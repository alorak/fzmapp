"""Opt-in AVESIS metadata collector; fails closed on robots restrictions."""
import json, re, time
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

rows = []
for slug in json.loads((ROOT/"data/profiles.json").read_text()):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",slug):
        raise ValueError("Invalid profile slug")
    page = get(BASE+slug+"/yayinlar")
    heading = page.find("h1")
    if not heading:
        raise RuntimeError("Profile heading missing")
    name = heading.get_text(" ",strip=True)
    seen = set()
    for a in page.select("a[href]"):
        url = urljoin(BASE,a["href"])
        if "/publication/details/" not in urlsplit(url).path.lower() or url in seen:
            continue
        title = re.sub(r"^\d+\.\s*","",a.get_text(" ",strip=True))
        if not title:
            continue
        seen.add(url)
        rows.append(dict(faculty=name,title=title,url=url,kind="publication"))
    if not seen:
        raise RuntimeError("No publication links found: "+slug)
target = ROOT/"data/academics.json"
target.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(f"Collected {len(rows)} records")
