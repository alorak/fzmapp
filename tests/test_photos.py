import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from scripts.photos import fetch_photo

PHOTO = "https://avesis.ankara.edu.tr/user/image/100112"


def png(size=(800, 1000)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (90, 110, 130)).save(buffer, "PNG")
    return buffer.getvalue()


def session_returning(response):
    robots = Mock(status_code=200, text="User-agent: *\nDisallow: /*/download*\n")
    session = Mock()
    session.get.side_effect = lambda url, **kwargs: robots if url.endswith("/robots.txt") else response
    return session


class PhotoTests(unittest.TestCase):
    def test_saves_resized_jpeg_after_redirect(self):
        response = Mock(status_code=200, url="https://cdn.example.org/p.png",
                        headers={"content-type": "image/png"}, content=png())
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "photos" / "mkabak.jpg"
            self.assertIsNone(fetch_photo(session_returning(response), PHOTO, target, {}))
            with Image.open(target) as saved:
                self.assertEqual(saved.format, "JPEG")
                self.assertLessEqual(saved.size[0], 400)
                self.assertLessEqual(saved.size[1], 500)

    def test_html_response_is_rejected(self):
        response = Mock(status_code=200, url="https://avesis.ankara.edu.tr/login",
                        headers={"content-type": "text/html"}, content=b"<html></html>")
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "photos" / "mkabak.jpg"
            error = fetch_photo(session_returning(response), PHOTO, target, {})
            self.assertIn("not an image", error)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
