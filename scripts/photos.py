"""Download the AVESIS profile photos listed in data/faculty.json into photos/<profile>.jpg."""

import io
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from PIL import Image
from protego import Protego

AGENT = "FZMAppResearchMetadata/0.1"
ROOT = Path(__file__).resolve().parents[1]
PROFILE = re.compile(r"[a-zA-Z0-9_.-]+")
MAX_BYTES = 5_000_000
SIZE = (400, 500)


def allowed(session, url, cache):
    """Check robots.txt of the URL's host, as the collector does."""
    origin = "{0.scheme}://{0.netloc}".format(urlsplit(url))
    if origin not in cache:
        response = session.get(urljoin(origin, "/robots.txt"), timeout=20, allow_redirects=False)
        cache[origin] = Protego.parse(response.text if response.status_code == 200 else "")
    return cache[origin].can_fetch(url, AGENT)


def fetch_photo(session, url, target, robots):
    """Save one photo as a JPEG. Returns an error message, or None on success."""
    if urlsplit(url).scheme != "https":
        return "not an https URL"
    if not allowed(session, url, robots):
        return "not permitted by robots.txt"
    response = session.get(url, timeout=30, allow_redirects=True)
    final = response.url or url
    if urlsplit(final).scheme != "https":
        return f"redirected to a non-https URL: {final}"
    if response.status_code != 200:
        return f"HTTP {response.status_code} from {final}"
    kind = response.headers.get("content-type", "")
    if not kind.startswith("image/"):
        return f"not an image ({kind or 'no content type'}) from {final}"
    if len(response.content) > MAX_BYTES:
        return "image too large"
    try:
        image = Image.open(io.BytesIO(response.content))
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        return f"unreadable image: {exc}"
    image = image.convert("RGB")
    image.thumbnail(SIZE)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "JPEG", quality=85, optimize=True)
    return None


def main():
    people = json.loads((ROOT / "data/faculty.json").read_text(encoding="utf-8"))
    saved, failed, robots = 0, [], {}
    with requests.Session() as session:
        session.headers["User-Agent"] = AGENT
        for person in people:
            profile, url = person.get("profile", ""), person.get("photo")
            if not url or not PROFILE.fullmatch(profile):
                continue
            try:
                error = fetch_photo(session, url, ROOT / "photos" / f"{profile}.jpg", robots)
            except requests.RequestException as exc:
                error = str(exc)
            if error:
                failed.append(profile)
                print(f"::warning::{profile}: {error}", flush=True)
            else:
                saved += 1
                print(f"{profile}: saved", flush=True)
            time.sleep(2)
    print(f"Saved {saved} photos; {len(failed)} failed: {', '.join(failed) or '-'}")
    if not saved:
        sys.exit(1)


if __name__ == "__main__":
    main()
