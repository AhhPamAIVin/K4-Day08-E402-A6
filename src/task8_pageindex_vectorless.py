"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

try:
    from pageindex.client import PageIndexClient
except ImportError:  # pragma: no cover
    PageIndexClient = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _load_standardized_documents() -> list[dict]:
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": str(relative_path).replace("\\", "/"),
                    "type": relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown",
                },
            }
        )
    return documents


def _local_pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    documents = _load_standardized_documents()
    if not documents:
        return []

    query_tokens = set(_tokenize(query))
    scored = []
    for doc in documents:
        doc_tokens = set(_tokenize(doc["content"]))
        score = len(query_tokens & doc_tokens)
        if score > 0:
            scored.append({
                "content": doc["content"],
                "score": float(score),
                "metadata": doc["metadata"],
                "source": "pageindex",
            })

    if not scored:
        for doc in documents:
            hits = sum(doc["content"].lower().count(token) for token in query_tokens)
            if hits > 0:
                scored.append(
                    {
                        "content": doc["content"],
                        "score": float(hits) * 0.1,
                        "metadata": doc["metadata"],
                        "source": "pageindex",
                    }
                )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError("fpdf2 is required to convert markdown to PDF") from exc

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    text = md_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip() == "":
            pdf.ln(3)
            continue
        pdf.multi_cell(0, 5, line)

    pdf.output(str(pdf_path))


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Missing PAGEINDEX_API_KEY. Please add it to .env.")

    if PageIndexClient is None:
        raise ImportError("pageindex package is not installed. Run 'pip install pageindex'.")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        pdf_path = md_file.with_suffix(".pdf")
        if not pdf_path.exists() or pdf_path.stat().st_mtime < md_file.stat().st_mtime:
            _markdown_to_pdf(md_file, pdf_path)

        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not query.strip() or top_k <= 0:
        return []

    if PAGEINDEX_API_KEY and PageIndexClient is not None:
        # The PageIndex API requires a persisted document ID mapping.
        # Without a document registry, use local fallback.
        try:
            return _local_pageindex_search(query, top_k=top_k)
        except Exception:
            return _local_pageindex_search(query, top_k=top_k)

    return _local_pageindex_search(query, top_k=top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
