import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from bs4 import BeautifulSoup

from scripts.collect import AGENT, BASE, PublicSite, collect, extract_links, extract_year

PROFILE = BASE + "mkabak/yayinlar"
ARTICLE = "1b6a8ec0-a865-4fc3-b2f4-fc7af4e1976a/normally-off-algan-gan-mis-hemt-with-low-gate-leakage-current-using-a-hydrofluoric-acid-pre-treatment"
TITLE = "Normally-off AlGaN/GaN MIS-HEMT with low gate leakage current using a hydrofluoric acid pre-treatment"
TURKISH_ARTICLE = f"/yayin/{ARTICLE}"
ENGLISH_ARTICLE = f"/publication/details/{ARTICLE}"
THESIS = "/yonetilen-tez/0a2d1528-8508-4ddc-865b-d542a81ebe5f/lazer-mesafe-olcer-optik-tasarimi"


def html(markup):
    return BeautifulSoup(markup, "html.parser")


class LinkTests(unittest.TestCase):
    def test_publications_in_both_languages(self):
        # Abridged from the public Mehmet Kabak listings with auxiliary links.
        for path in (TURKISH_ARTICLE, ENGLISH_ARTICLE):
            with self.subTest(path=path):
                page = html(f"""
                    <h3 class="title"><a target="_blank" href="{path}">
                        <span>1.</span><strong>{TITLE}</strong>
                    </a></h3>
                    <a href="{path}" aria-label="Abstract"><i class="fa fa-book"></i></a>
                    <a href="{path}/download">Download</a>
                    <a href="/publication/showdocument/1b6a8ec0-a865-4fc3-b2f4-fc7af4e1976a">PDF</a>
                    <a href="/mkabak/yayinlar">Publications navigation</a>
                    <a href="https://example.org{path}">External lookalike</a>
                    <a href="http://avesis.ankara.edu.tr{path}">Insecure lookalike</a>
                """)
                self.assertEqual(
                    list(extract_links(page, "publication", PROFILE)),
                    [(BASE.rstrip("/") + path, TITLE)],
                )

    def test_turkish_thesis_detail(self):
        page = html(f"""
            <a href="{THESIS}#detail">LAZER MESAFE ÖLÇER OPTİK TASARIMI</a>
            <a href="{THESIS}"><i class="fa fa-book"></i></a>
            <a href="/mkabak/deneyim">Akademik İdari Deneyim</a>
        """)
        self.assertEqual(
            list(extract_links(page, "thesis", BASE + "mkabak/deneyim")),
            [(BASE.rstrip("/") + THESIS, "LAZER MESAFE ÖLÇER OPTİK TASARIMI")],
        )


class CollectionTests(unittest.TestCase):
    def test_collector_writes_unique_rows_and_metadata(self):
        article_url = BASE.rstrip("/") + TURKISH_ARTICLE
        thesis_url = BASE.rstrip("/") + THESIS
        pages = {
            PROFILE: html(f"""
                <h1>Prof. Dr. MEHMET KABAK</h1>
                <img src="/Content/images/logo.png">
                <img src="/user/image/100112" alt="Profile photo">
                <a href="{TURKISH_ARTICLE}">1. {TITLE}</a>
                <a href="{TURKISH_ARTICLE}#detail">{TITLE}</a>
            """),
            BASE + "mkabak/deneyim": html(f'<a href="{THESIS}">Thesis title</a>'),
            article_url: html("""
                <li><strong>Doi Numarası:</strong>
                    <a href="http://doi.org/10.1016/j.sse.2019.05.008">10.1016/j.sse.2019.05.008</a>
                </li>
                <li><strong>Anahtar Kelimeler:</strong> GaN, Gate leakage current, HEMT</li>
            """),
            thesis_url: html("<h1>Thesis detail</h1>"),
        }
        site = Mock()
        site.get.side_effect = pages.__getitem__
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp)
            collect(["mkabak"], site, output)
            rows = json.loads((output / "academics.json").read_text(encoding="utf-8"))
            report = json.loads((output / "collection-report.json").read_text(encoding="utf-8"))
            people = json.loads((output / "faculty.json").read_text(encoding="utf-8"))

        self.assertEqual([row["kind"] for row in rows], ["publication", "thesis"])
        self.assertEqual(rows[0]["title"], TITLE)
        self.assertEqual(rows[0]["doi"], "10.1016/j.sse.2019.05.008")
        self.assertEqual(rows[0]["keywords"], "GaN, Gate leakage current, HEMT")
        self.assertEqual(report[0]["counts"], {"publication": 1, "thesis": 1})
        self.assertEqual(report[0]["status"], "ok")
        self.assertEqual(site.get.call_count, 4)
        self.assertEqual(people, [{
            "faculty": "Prof. Dr. MEHMET KABAK",
            "profile": "mkabak",
            "profile_url": BASE + "mkabak",
            "photo": BASE + "user/image/100112",
        }])


class YearTests(unittest.TestCase):
    def test_publication_and_thesis_years(self):
        cases = {
            "<ul><li><strong>Basım Tarihi:</strong> 2026</li></ul>": 2026,
            "<p><b>Tezin Onay Tarihi:</b></p><p>2018</p>": 2018,
            "<li>Publication Date: March 2019</li>": 2019,
            "<li>Cilt: 2021-5</li>": None,
        }
        for markup, year in cases.items():
            with self.subTest(markup=markup):
                self.assertEqual(extract_year(html(markup)), year)


class PublicSiteTests(unittest.TestCase):
    def test_wildcard_disallow_and_rate_limit_are_enforced(self):
        session = Mock()
        session.get.side_effect = [
            Mock(status_code=200, text="User-agent: *\nDisallow: /*/download*\nCrawl-delay: 3\n"),
            Mock(status_code=429),
        ]
        site = PublicSite(BASE, session)
        self.assertEqual(site.delay, 3)
        self.assertFalse(site.robots.can_fetch(BASE + "mkabak/download?languageCode=tr", AGENT))
        with self.assertRaisesRegex(RuntimeError, "not permitted by origin or robots"):
            site.get(BASE + "mkabak/download?languageCode=tr")
        with self.assertRaisesRegex(RuntimeError, "not permitted by origin or robots"):
            site.get("https://example.org/" + ARTICLE)
        with patch("scripts.collect.time.sleep"), self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            site.get(PROFILE)
        with self.assertRaisesRegex(RuntimeError, "skipped after HTTP 429"):
            site.get(BASE + "mkabak/deneyim")
        self.assertEqual(session.get.call_args_list, [
            call(BASE + "robots.txt", timeout=20, allow_redirects=False),
            call(PROFILE, timeout=30, allow_redirects=False),
        ])

    def test_unavailable_robots_fail_closed(self):
        session = Mock()
        session.get.return_value = Mock(status_code=403)
        with self.assertRaisesRegex(RuntimeError, "robots.txt unavailable"):
            PublicSite(BASE, session)
        session.get.assert_called_once_with(BASE + "robots.txt", timeout=20, allow_redirects=False)


if __name__ == "__main__":
    unittest.main()
