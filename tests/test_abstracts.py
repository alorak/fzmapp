import unittest

from bs4 import BeautifulSoup

from scripts.abstracts import extract_abstract

ABSTRACT = "Bu tez çalışmasında NiCoMnSn tabanlı manyetik şekil hafıza alaşımlarında martensitik geçiş ve manyetokalorik etkinin araştırılması hedeflenmiştir."


class AbstractTests(unittest.TestCase):
    def test_label_followed_by_sibling_block(self):
        page = BeautifulSoup(f"""
            <p><strong>Danışman:</strong> Barış Emre</p>
            <p><strong>Özet:</strong></p>
            <p>{ABSTRACT}</p>
        """, "html.parser")
        self.assertEqual(extract_abstract(page), ABSTRACT)

    def test_label_inside_same_block(self):
        page = BeautifulSoup(f"<div><strong>Özet:</strong> {ABSTRACT}</div>", "html.parser")
        self.assertEqual(extract_abstract(page), ABSTRACT)

    def test_navigation_tab_is_not_the_label(self):
        page = BeautifulSoup(f"""
            <nav><span>Özet</span><a>Yayınlar</a><a>Projeler</a><a>Dersler</a><a>Tezler</a><a>Akademik ve İdari Deneyim</a><a>İletişim</a></nav>
            <p><strong>Özet:</strong></p><p>{ABSTRACT}</p>
        """, "html.parser")
        self.assertEqual(extract_abstract(page), ABSTRACT)

    def test_page_without_abstract(self):
        page = BeautifulSoup("<p><strong>Tezin Türü:</strong> Yüksek Lisans</p>", "html.parser")
        self.assertIsNone(extract_abstract(page))


if __name__ == "__main__":
    unittest.main()
