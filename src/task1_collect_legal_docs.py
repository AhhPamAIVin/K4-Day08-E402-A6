"""Task 1: collect official Shopee policies as DOCX files."""

from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

POLICIES = [
    {
        "url": "https://help.shopee.vn/portal/4/article/77251?seo=1",
        "filename": "shopee-return-refund-policy.docx",
        "customer_role": "both",
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/77243?seo=1",
        "filename": "shopee-terms-of-service.docx",
        "customer_role": "both",
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/77244?seo=1",
        "filename": "shopee-privacy-policy.docx",
        "customer_role": "both",
    },
]


class _TextExtractor(HTMLParser):
    """Small dependency-free extractor for server-rendered help pages."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "svg", "noscript"}:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if not self._skip and tag in {"p", "div", "li", "h1", "h2", "h3", "tr", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "svg", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip:
            return
        text = " ".join(unescape(data).split())
        if text:
            if self._in_title:
                self.title += text
            self.parts.append(text + " ")

    def content(self):
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Data directory: {DATA_DIR}")


def fetch_page(url: str) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (RAG course project)"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(html)
    content = parser.content()
    if len(content) < 500:
        raise ValueError(f"Trang không có đủ nội dung: {url}")
    return parser.title.split("|")[0].strip() or "Shopee Policy", content


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_docx(path: Path, title: str, source_url: str, customer_role: str, content: str):
    """Write a minimal standards-compliant DOCX without an extra dependency."""
    paragraphs = [title, f"Source: {source_url}", f"customer_role: {customer_role}", "", *content.splitlines()]
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{_xml_escape(p)}</w:t></w:r></w:p>'
        for p in paragraphs if p is not None
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)


def collect_policies():
    setup_directory()
    for index, policy in enumerate(POLICIES, 1):
        print(f"[{index}/{len(POLICIES)}] Downloading: {policy['url']}")
        title, content = fetch_page(policy["url"])
        output = DATA_DIR / policy["filename"]
        write_docx(output, title, policy["url"], policy["customer_role"], content)
        print(f"  [OK] Saved: {output}")


if __name__ == "__main__":
    collect_policies()
