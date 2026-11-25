"""Collect public AVESIS metadata on demand, subject to robots.txt."""

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from protego import Protego

BASE = "https://avesis.ankara.edu.tr/"
AGENT = "FZMAppResearchMetadata/0.1"
ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_PATH = re.compile(
    r"^/(?:yayin|publication/details)/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?:/[^/]+)?/?$",
    re.IGNORECASE,
)
THESIS_PATH = re.compile(r"/(?:yonetilen-tez|thesis|theses|advising|supervisedthesis)/[^/]+", re.I)


class PublicSite:
    def __init__(self, base, session):
        self.origin = (urlsplit(base).scheme, urlsplit(base).netloc)
        self.session = session
        robots_url = urljoin(base, "/robots.txt")
        response = session.get(robots_url, timeout=20, allow_redirects=False)
        if response.status_code != 200 or "<html" in response.text.lower():
            raise RuntimeError(f"robots.txt unavailable (HTTP {response.status_code}): {robots_url}")
        self.robots = Protego.parse(response.text)
        rate = self.robots.request_rate(AGENT)
        self.delay = max(2, self.robots.crawl_delay(AGENT) or 0, rate.seconds / rate.requests if rate else 0)
        self.next_request = time.monotonic() + self.delay
        self.rate_limited = False

    def get(self, url):
        parts = urlsplit(url)
        if (parts.scheme, parts.netloc) != self.origin or not self.robots.can_fetch(url, AGENT):
            raise RuntimeError(f"URL not permitted by origin or robots.txt: {url}")
        if self.rate_limited:
            raise RuntimeError(f"Request skipped after HTTP 429: {url}")
        time.sleep(max(0, self.next_request - time.monotonic()))
        self.next_request = time.monotonic() + self.delay
        response = self.session.get(url, timeout=30, allow_redirects=False)
        if response.status_code == 429:
            self.rate_limited = True
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {url}")
        return BeautifulSoup(response.text, "html.parser")


def extract_links(page, kind, listing_url):
    for anchor in page.select("a[href]"):
        url = urldefrag(urljoin(listing_url, anchor["href"])).url
        location = urlsplit(url)
        if location.scheme != "https" or location.netloc != urlsplit(BASE).netloc:
            continue
        if kind == "publication" and not PUBLICATION_PATH.fullmatch(location.path):
            continue
        if kind == "thesis" and not THESIS_PATH.search(location.path):
            continue
        title = re.sub(r"^\d+\.\s*", "", anchor.get_text(" ", strip=True))
        if title:
            yield url, title


def metadata(page):
    result = {}
    for anchor in page.select("a[href]"):
        url = urlsplit(anchor["href"])
        if url.hostname in ("doi.org", "dx.doi.org"):
            result["doi"] = url.path.lstrip("/")
            break
    for item in page.select("li"):
        value = item.get_text(" ", strip=True)
        match = re.fullmatch(r"(?:Publication Date|Yayın Tarihi)\s*:\s*(\d{4})", value, re.I)
        if match:
            result["year"] = int(match[1])
        if re.match(r"^(Keywords|Anahtar Kelimeler)\s*:", value, re.I):
            result["keywords"] = value.split(":", 1)[1].strip()
    # Abstract redistribution is disabled until reuse rights are confirmed.
    return result


def discover(session):
    profiles = set(json.loads((ROOT / "data/profiles.json").read_text(encoding="utf-8")))
    if os.getenv("DISCOVER_DEPARTMENT", "false").lower() != "true":
        if not profiles:
            raise RuntimeError("Add AVESIS profile slugs to data/profiles.json")
        print(f"Manual profile list: {len(profiles)} profiles; department discovery disabled")
        return sorted(profiles)

    source = "https://phys.eng.ankara.edu.tr"
    department = PublicSite(source, session)
    for path in ("/tr/ogretim-uyeleri/", "/tr/personel/akademik-son/"):
        for anchor in department.get(source + path).select("a[href]"):
            url = urlsplit(anchor["href"])
            if url.hostname == "avesis.ankara.edu.tr":
                slug = url.path.strip("/").split("/")[0]
                if re.fullmatch(r"[a-zA-Z0-9_.-]+", slug):
                    profiles.add(slug)
    if len(profiles) <= 1:
        raise RuntimeError("Department discovery returned no additional profiles")
    return sorted(profiles)


def collect(profiles, site, output_dir):
    rows, report, cache = [], [], {}
    for slug in profiles:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", slug):
            raise ValueError("Invalid profile slug")
        entry = dict(profile=slug, counts={"publication": 0, "thesis": 0}, errors=[])
        report.append(entry)
        print(f"Collecting {slug}", flush=True)
        publication_url = f"{BASE}{slug}/yayinlar"
        try:
            page = site.get(publication_url)
            heading = page.find("h1")
            if not heading:
                raise RuntimeError("Profile heading missing")
            name = heading.get_text(" ", strip=True)
            entry["faculty"] = name
        except (RuntimeError, requests.RequestException) as exc:
            entry["errors"].append(str(exc))
            entry["status"] = "failed"
            print(f"::warning::{slug}: {exc}", flush=True)
            continue

        seen = set()
        for kind, listing_url in (("publication", publication_url), ("thesis", f"{BASE}{slug}/deneyim")):
            try:
                listing = page if kind == "publication" else site.get(listing_url)
                links = list(extract_links(listing, kind, listing_url))
                if not links and kind == "publication":
                    page_title = listing.title.get_text(" ", strip=True) if listing.title else "(missing)"
                    anchor_count = len(listing.select("a[href]"))
                    raise RuntimeError(f"No publication links found; page title: {page_title}; anchor count: {anchor_count}")
                for url, title in links:
                    if url in seen:
                        continue
                    seen.add(url)
                    if url not in cache:
                        try:
                            cache[url] = metadata(site.get(url)), None
                        except (RuntimeError, requests.RequestException) as exc:
                            cache[url] = {}, str(exc)
                    fields, error = cache[url]
                    if error:
                        entry["errors"].append(error)
                    rows.append(dict(faculty=name, title=title, url=url, kind=kind, **fields))
                    entry["counts"][kind] += 1
            except (RuntimeError, requests.RequestException) as exc:
                entry["errors"].append(str(exc))
                print(f"::warning::{slug}: {exc}", flush=True)
        entry["status"] = "partial" if entry["errors"] else "ok"
        counts = entry["counts"]
        print(f"{slug}: {counts['publication']} publications, {counts['thesis']} theses, {len(entry['errors'])} errors", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "collection-report.json", report)
    if not rows:
        raise RuntimeError("No records collected; inspect collection-report.json")
    write_json(output_dir / "academics.json", rows)
    print(f"Collected {len(rows)} records; inspect report for partial coverage")


def write_json(target, data):
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.getenv("AVESIS_PROFILE"), help="collect one configured profile")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    with requests.Session() as session:
        session.headers["User-Agent"] = AGENT
        profiles = discover(session)
        if args.profile:
            if args.profile not in profiles:
                parser.error(f"unknown AVESIS profile: {args.profile}")
            profiles = [args.profile]
        collect(profiles, PublicSite(BASE, session), args.output_dir)


if __name__ == "__main__":
    main()
