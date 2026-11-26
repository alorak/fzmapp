"""Collect the publication year of every record from its AVESIS page into data/years.json.

The full collector (collect.py) now stores the year in academics.json itself; this script fills the years
for the current data without re-collecting everything.
"""

import json
import re

import requests

from scripts.collect import AGENT, BASE, ROOT, PublicSite, extract_year, write_json

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def main():
    records = json.loads((ROOT / "data/academics.json").read_text(encoding="utf-8"))
    target = ROOT / "data/years.json"
    years = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    missing = []
    with requests.Session() as session:
        session.headers["User-Agent"] = AGENT
        site = PublicSite(BASE, session)
        for number, record in enumerate(records, 1):
            match = UUID.search(record.get("url", ""))
            if not match or match[0].lower() in years:
                continue
            key = match[0].lower()
            try:
                year = extract_year(site.get(record["url"]))
            except (RuntimeError, requests.RequestException) as exc:
                print(f"::warning::{record['url']}: {exc}", flush=True)
                if site.rate_limited:
                    break
                year = None
            if year:
                years[key] = year
            else:
                missing.append(record["url"])
            if number % 50 == 0:
                print(f"{number}/{len(records)}: {len(years)} years", flush=True)
                write_json(target, years)
    write_json(target, years)
    print(f"Saved {len(years)} years; {len(missing)} pages without a year")
    for url in missing[:20]:
        print("missing:", url)


if __name__ == "__main__":
    main()

