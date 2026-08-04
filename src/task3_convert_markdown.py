"""Task 3: normalize landing DOCX/PDF/JSON files to Markdown."""

import json
import re
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _docx_fallback(filepath: Path) -> str:
    with ZipFile(filepath) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines = []
    for paragraph in root.iter(namespace + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(namespace + "t")).strip()
        if text:
            lines.append(text)
    return "\n\n".join(lines)


def _convert_document(filepath: Path) -> str:
    # DOCX is OpenXML, so this built-in parser avoids requiring MarkItDown's
    # separate [docx] extra. PDF still goes through Microsoft MarkItDown.
    if filepath.suffix.lower() == ".docx":
        return _docx_fallback(filepath)
    try:
        from markitdown import MarkItDown
        return MarkItDown().convert(str(filepath)).text_content
    except ModuleNotFoundError:
        raise RuntimeError('Hãy cài dependency bằng: pip install "markitdown[pdf]"')


def convert_legal_docs():
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue
        print(f"Converting: {filepath.name}")
        content = _convert_document(filepath).strip()
        if len(content) < 200:
            raise ValueError(f"Nội dung sau convert quá ngắn: {filepath}")
        (output_dir / f"{filepath.stem}.md").write_text(content, encoding="utf-8")


def convert_news_articles():
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filepath in sorted(news_dir.glob("*.json")):
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        header = (
            f"# {data.get('title', 'Unknown')}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n"
            f"**customer_role:** {data.get('customer_role', 'both')}\n\n---\n\n"
        )
        body = data.get("content_markdown") or data.get("content", "")
        body = re.sub(r"^# .+?\n+", "", body, count=1)
        content = header + body.strip()
        if len(content) < 200:
            raise ValueError(f"Nội dung JSON quá ngắn: {filepath}")
        (output_dir / f"{filepath.stem}.md").write_text(content, encoding="utf-8")


def convert_all():
    print("Task 3: Convert to Markdown")
    convert_legal_docs()
    convert_news_articles()
    print(f"[OK] Done: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
