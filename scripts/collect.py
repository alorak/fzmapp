"""Opt-in AVESIS metadata collector; fails closed on robots restrictions."""
import json, re, time, os
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

BASE = "https://avesis.ankara.edu.tr/"
AGENT = "FZMAppResearchMetadata/0.1"
ROOT = Path(__file__).resolve().parents[1]
session = requests.Session()
session.headers["User-Agent"] = AGENT
robots = RobotFileParser()
reply = session.get(BASE+"robots.txt", timeout=20, allow_redirects=False)
reply.raise_for_status()
if reply.status_code != 200 or "<html" in reply.text.lower():
    raise RuntimeError("robots.txt unavailable; aborting")
robots.parse(reply.text.splitlines())

def get(url):
    if urlsplit(url).netloc != urlsplit(BASE).netloc or not robots.can_fetch(AGENT,url):
        raise RuntimeError("URL not permitted: "+url)
    time.sleep(max(2, robots.crawl_delay(AGENT) or 0))
    response = session.get(url, timeout=30, allow_redirects=False)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {url}")
    return BeautifulSoup(response.text,"html.parser")

def extract_links(page, kind):
    for a in page.select("a[href]"):
        url = urljoin(BASE, a["href"])
        path = urlsplit(url).path.lower()
        if urlsplit(url).netloc != urlsplit(BASE).netloc:
            continue
        if kind == "publication" and "/publication/details/" not in path:
            continue
        if kind == "thesis" and not re.search(r"/(?:thesis|theses|advising|supervisedthesis)/", path):
            continue
        title = re.sub(r"^\d+\.\s*", "", a.get_text(" ", strip=True))
        if title:
            yield url, title

def metadata(page):
    result = {}
    for a in page.select("a[href]"):
        url = urlsplit(a["href"])
        if url.hostname in ("doi.org", "dx.doi.org"):
            result["doi"] = url.path.lstrip("/")
            break
    for li in page.select("li"):
        value = li.get_text(" ", strip=True)
        match = re.fullmatch(r"(?:Publication Date|Yayın Tarihi)\s*:\s*(\d{4})", value, re.I)
        if match:
            result["year"] = int(match[1])
        if re.match(r"^(Keywords|Anahtar Kelimeler)\s*:", value, re.I):
            result["keywords"] = value.split(":", 1)[1].strip()
    # Abstract redistribution is disabled until reuse rights are confirmed.
    return result

def discover():
    profiles = set(json.loads((ROOT/"data/profiles.json").read_text()))
    if os.getenv("DISCOVER_DEPARTMENT", "false").lower() != "true":
        if not profiles:
            raise RuntimeError("Add AVESIS profile slugs to data/profiles.json")
        print(f"Manual profile list: {len(profiles)} profiles; department discovery disabled")
        return sorted(profiles)
    source = "https://phys.eng.ankara.edu.tr"
    rp = RobotFileParser()
    resp = session.get(source+"/robots.txt", timeout=20, allow_redirects=False)
    if resp.status_code != 200 or "<html" in resp.text.lower():
        raise RuntimeError("Department robots unavailable")
    rp.parse(resp.text.splitlines())
    for path in ("/tr/ogretim-uyeleri/", "/tr/personel/akademik-son/"):
        url = source+path
        if not rp.can_fetch(AGENT,url):
            raise RuntimeError("Department discovery not permitted by robots.txt: "+url+"; run with discovery disabled and supply profiles.json")
        time.sleep(max(2,rp.crawl_delay(AGENT) or 0))
        resp = session.get(url,timeout=30,allow_redirects=False)
        if resp.status_code != 200:
            raise RuntimeError("Department page unavailable")
        soup = BeautifulSoup(resp.text,"html.parser")
        for a in soup.select("a[href]"):
            u = urlsplit(a["href"])
            if u.hostname == "avesis.ankara.edu.tr":
                slug = u.path.strip("/").split("/")[0]
                if re.fullmatch(r"[a-zA-Z0-9_.-]+",slug):
                    profiles.add(slug)
    if len(profiles) <= 1:
        raise RuntimeError('Department discovery returned no additional profiles')
    return sorted(profiles)

rows = []
report = []
cache = {}
for slug in discover():
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", slug):
        raise ValueError("Invalid profile slug")
    entry = dict(profile=slug, counts={"publication": 0, "thesis": 0}, errors=[])
    report.append(entry)
    print("Collecting "+slug, flush=True)
    try:
        page = get(BASE+slug+"/yayinlar")
        heading = page.find("h1")
        if not heading:
            raise RuntimeError("Profile heading missing")
        name = heading.get_text(" ", strip=True)
        entry["faculty"] = name
    except (RuntimeError, requests.RequestException) as exc:
        entry["errors"].append(str(exc))
        entry["status"] = "failed"
        print("::warning::"+slug+": "+str(exc), flush=True)
        continue
    seen = set()
    for kind, suffix in (("publication", "/yayinlar"), ("thesis", "/deneyim")):
        try:
            listing = page if kind == "publication" else get(BASE+slug+suffix)
            links = list(extract_links(listing, kind))
            if not links and kind == "publication":
                raise RuntimeError("No publication links found; page title: "+str(listing.title.get_text(" ",strip=True) if listing.title else "(missing)")+"; anchor count: "+str(len(listing.select("a[href]"))))
            for url, title in links:
                if url in seen:
                    continue
                seen.add(url)
                if url not in cache:
                    try:
                        cache[url] = metadata(get(url))
                    except (RuntimeError, requests.RequestException) as exc:
                        entry["errors"].append(str(exc))
                        cache[url] = {}
                rows.append(dict(faculty=name,title=title,url=url,kind=kind,**cache[url]))
                entry["counts"][kind] += 1
        except (RuntimeError, requests.RequestException) as exc:
            entry["errors"].append(str(exc))
            print("::warning::"+slug+": "+str(exc), flush=True)
    entry["status"] = "partial" if entry["errors"] else "ok"
target = ROOT/"data/academics.json"
tmp = target.with_suffix(".tmp")
tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(ROOT/"data/collection-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if not rows:
    tmp.unlink(missing_ok=True)
    raise RuntimeError("No records collected; inspect collection-report.json")
tmp.replace(target)
print(f"Collected {len(rows)} records; inspect report for partial coverage")
