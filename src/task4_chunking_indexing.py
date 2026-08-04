"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model:
    - OpenAI text-embedding-3-small (1536 dim), gọi qua API và không tải model local.
    - Cần OPENAI_API_KEY trong file .env.

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters chromadb openai

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter với size 800 / overlap 100.
# - 800 ký tự ~ 200-250 token tiếng Việt: đủ chứa trọn 1 điều khoản chính sách
#   (mỗi mục trong tài liệu Shopee thường 3-6 câu) mà không nhồi 2 chủ đề vào 1 chunk.
# - overlap 100 (12.5%) giữ lại câu bắc cầu ở ranh giới chunk, tránh mất ngữ cảnh
#   khi 1 quy định bị cắt ngang giữa chừng.
# - "recursive" thay vì "markdown_header": corpus có cả PDF convert (heading không
#   đều) lẫn news crawl, tách theo heading sẽ ra chunk dài ngắn rất lệch.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding: OpenAI text-embedding-3-small (1536 dim) — chạy qua API, không tải model
# local (máy lab không đủ RAM cho bge-m3), hỗ trợ tiếng Việt tốt, giá rẻ.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

# Vector store: ChromaDB — local persistent, không cần Docker, có sẵn cosine space.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "ecommerce_support_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # Reference implementation outline:
    # documents = []
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     content = md_file.read_text(encoding="utf-8")
    #     doc_type = "legal" if "legal" in str(md_file) else "news"
    #     documents.append({
    #         "content": content,
    #         "metadata": {"source": md_file.name, "type": doc_type}
    #     })
    # return documents
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        role_match = re.search(r"\*\*customer_role:\*\*\s*(buyer|seller|both)", content, re.I)
        documents.append({
            "content": content,
            "metadata": {
                "source": str(relative_path).replace("\\", "/"),
                "type": relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown",
                "customer_role": role_match.group(1).lower() if role_match else "both",
            },
        })
    return documents


def is_useful_chunk(text: str) -> bool:
    """Loại chunk rác từ trang crawl: menu, footer, danh sách link/ảnh.

    Trang news crawl về kèm nhiều điều hướng (`[text](url)`, `![img](url)`). Những
    chunk này không mang thông tin chính sách nhưng vẫn được embed và có thể lọt vào
    top-k, làm nhiễu context gửi cho LLM ở Task 10.
    """
    import re

    stripped = text.strip()
    if len(stripped) < 80:
        return False

    markup_chars = sum(len(m) for m in re.findall(r"!?\[[^\]]*\]\([^)]*\)", stripped))
    return markup_chars / len(stripped) < 0.5


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # Reference implementation outline:
    #
    # Ví dụ với RecursiveCharacterTextSplitter:
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    #
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP,
    #     separators=["\n\n", "\n", ". ", " ", ""]
    # )
    # chunks = []
    # for doc in documents:
    #     splits = splitter.split_text(doc["content"])
    #     for i, chunk_text in enumerate(splits):
    #         chunks.append({
    #             "content": chunk_text,
    #             "metadata": {**doc["metadata"], "chunk_index": i}
    #         })
    # return chunks
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        index = 0
        for content in splitter.split_text(doc["content"]):
            if not is_useful_chunk(content):
                continue
            chunks.append({
                "content": content,
                "metadata": {**doc["metadata"], "chunk_index": index},
            })
            index += 1
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo embedding bằng OpenAI API; không tải hoặc chạy model local."""
    if EMBEDDING_PROVIDER != "openai":
        raise ValueError("Project này chỉ hỗ trợ EMBEDDING_PROVIDER=openai")
    if not texts:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Thiếu OPENAI_API_KEY. Hãy copy .env.example thành .env và điền API key.")

    from openai import OpenAI

    client = OpenAI()
    embeddings = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start:start + batch_size],
        )
        embeddings.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
    return embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def get_collection():
    """Mở collection ChromaDB dùng chung cho indexing và search."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    # Reference implementation outline:
    #
    # Ví dụ với ChromaDB:
    # import chromadb
    #
    # CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # collection = client.get_or_create_collection(
    #     name=COLLECTION_NAME,
    #     metadata={"hnsw:space": "cosine"},
    # )
    #
    # ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    # collection.upsert(
    #     ids=ids,
    #     documents=[c["content"] for c in chunks],
    #     embeddings=[c["embedding"] for c in chunks],
    #     metadatas=[c["metadata"] for c in chunks],
    # )
    if not chunks:
        return
    collection = get_collection()
    ids = [
        f"{chunk['metadata']['source']}_chunk_{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
