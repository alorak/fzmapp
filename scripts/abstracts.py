"""Collect the abstracts of supervised theses from their public AVESIS pages into data/abstracts.json.

Only theses are collected: they are Ankara University's own works shown in its open access collection.
Publication abstracts belong to their publishers and are fetched on demand in the browser instead.
"""

import json
import re

import requests
from bs4 import BeautifulSoup

from scripts.collect import BASE, AGENT, ROOT, PublicSite, write_json

# Labels with a colon are tried first so that a navigation tab named "Özet" is not mistaken for the label.
LABELS = (("özet:", "abstract:"), ("özet", "abstract"))
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
MAX_CHARS = 6000


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_abstract(page):
    """Return the text after the "Özet:" label of an AVESIS thesis page, or None."""
    candidates = page.find_all(["strong", "b", "label", "span", "dt", "h2", "h3", "h4", "h5", "p", "div"])
    for labels in LABELS:
        for label in candidates:
            if clean(label.get_text(" ")).lower() not in labels:
                continue
            text = text_after(label)
            if text:
                return text
    return None


def text_after(label):
    # The abstract is either the rest of the label's parent or the element that follows it.
    parent_text = clean(label.parent.get_text(" "))
    own = clean(label.get_text(" "))
    rest = clean(parent_text[parent_text.find(own) + len(own):]) if own in parent_text else ""
    if len(rest) < 80:
        following = label.find_next_sibling() or label.parent.find_next_sibling()
        rest = clean(following.get_text(" ")) if following else ""
    return rest[:MAX_CHARS] if len(rest) >= 80 else None


def main():
    records = json.loads((ROOT / "data/academics.json").read_text(encoding="utf-8"))
    theses = [record for record in records if record.get("kind") == "thesis" and record.get("url")]
    found, missing = {}, []
    with requests.Session() as session:
        session.headers["User-Agent"] = AGENT
        site = PublicSite(BASE, session)
        for record in theses:
            match = UUID.search(record["url"])
            if not match:
                continue
            key = match[0].lower()
            if key in found:
                continue
            try:
                text = extract_abstract(site.get(record["url"]))
            except (RuntimeError, requests.RequestException) as exc:
                print(f"::warning::{record['url']}: {exc}", flush=True)
                text = None
            if text:
                found[key] = {"text": text, "source": "AVESİS"}
            else:
                missing.append(record["url"])
            print(f"{len(found)} found, {len(missing)} missing", flush=True)
    write_json(ROOT / "data/abstracts.json", found)
    print(f"Saved {len(found)} thesis abstracts; {len(missing)} without an abstract")
    for url in missing[:20]:
        print("missing:", url)


if __name__ == "__main__":
    main()
