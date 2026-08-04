"""Task 8: PageIndex vectorless retrieval with a local offline fallback."""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_API_URL = os.getenv("PAGEINDEX_API_URL", "https://api.pageindex.ai").rstrip("/")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = PROJECT_ROOT / "pageindex_doc_ids.json"
PAGEINDEX_PDF_DIR = PROJECT_ROOT / "pageindex_pdfs"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _load_documents() -> list[dict]:
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative = md_file.relative_to(STANDARDIZED_DIR)
        documents.append({
            "content": content,
            "metadata": {
                "source": str(relative).replace("\\", "/"),
                "type": relative.parts[0] if len(relative.parts) > 1 else "unknown",
            },
        })
    return documents


def _local_search(query: str, top_k: int) -> list[dict]:
    query_tokens = set(_tokenize(query))
    results = []
    for document in _load_documents():
        sections = re.split(r"(?m)(?=^#{1,6}\s+)", document["content"])
        for section_index, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            for offset in range(0, len(section), 4000):
                passage = section[offset:offset + 4000]
                tokens = set(_tokenize(passage))
                score = len(query_tokens & tokens) / max(1, len(query_tokens))
                if score:
                    metadata = dict(document["metadata"])
                    metadata["section_index"] = section_index
                    results.append({
                        "content": passage,
                        "metadata": metadata,
                        "score": float(score),
                        "source": "pageindex",
                    })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path:
        pdf.add_font("Unicode", fname=str(font_path))
        pdf.set_font("Unicode", size=10)
    else:
        raise RuntimeError("Không tìm thấy font Unicode để tạo PDF cho PageIndex")
    for line in md_path.read_text(encoding="utf-8").splitlines():
        pdf.multi_cell(0, 5, line or " ")
    pdf.output(str(pdf_path))


def _load_registry() -> dict[str, str]:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(registry: dict[str, str]) -> None:
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def upload_documents() -> dict[str, str]:
    """Upload standardized documents and persist source -> PageIndex doc_id."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Missing PAGEINDEX_API_KEY in .env")
    import requests

    registry = _load_registry()
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        source = str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/")
        if source in registry:
            continue
        pdf_path = (PAGEINDEX_PDF_DIR / source).with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        _markdown_to_pdf(md_file, pdf_path)
        with pdf_path.open("rb") as file:
            response = requests.post(
                f"{PAGEINDEX_API_URL}/doc/",
                headers={"api_key": PAGEINDEX_API_KEY},
                files={"file": (pdf_path.name, file, "application/pdf")},
                timeout=120,
            )
        response.raise_for_status()
        registry[source] = response.json()["doc_id"]
        _save_registry(registry)
    return registry


def _parse_nodes(payload: dict, source: str) -> list[dict]:
    results = []
    for rank, node in enumerate(payload.get("retrieved_nodes", []), 1):
        contents = node.get("relevant_contents", [])
        if contents and isinstance(contents[0], list):
            contents = [item for group in contents for item in group]
        for item in contents:
            content = item.get("relevant_content") or item.get("content") or ""
            if content.strip():
                results.append({
                    "content": content.strip(),
                    "score": 1.0 / rank,
                    "metadata": {
                        "source": source,
                        "type": source.split("/", 1)[0] if "/" in source else "unknown",
                        "section": item.get("section_title") or node.get("title", ""),
                        "page_index": item.get("page_index"),
                    },
                    "source": "pageindex",
                })
    return results


def _remote_search(query: str, top_k: int) -> list[dict]:
    import requests

    results = []
    for source, doc_id in _load_registry().items():
        response = requests.post(
            f"{PAGEINDEX_API_URL}/retrieval/",
            headers={"api_key": PAGEINDEX_API_KEY},
            json={"doc_id": doc_id, "query": query, "thinking": False},
            timeout=60,
        )
        response.raise_for_status()
        retrieval_id = response.json()["retrieval_id"]
        for _ in range(30):
            poll = requests.get(
                f"{PAGEINDEX_API_URL}/retrieval/{retrieval_id}/",
                headers={"api_key": PAGEINDEX_API_KEY},
                timeout=30,
            )
            poll.raise_for_status()
            payload = poll.json()
            if payload.get("status") == "completed":
                results.extend(_parse_nodes(payload, source))
                break
            if payload.get("status") in {"failed", "error"}:
                break
            time.sleep(2)
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip() or top_k <= 0:
        return []
    if PAGEINDEX_API_KEY and _load_registry():
        return _remote_search(query, top_k)
    return _local_search(query, top_k)


if __name__ == "__main__":
    upload_documents()
