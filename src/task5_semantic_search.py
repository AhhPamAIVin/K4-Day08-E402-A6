"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Bonus — HyDE (Hypothetical Document Embeddings):
    Query người dùng thường ngắn và dùng từ ngữ khác với văn bản chính sách gốc
    (vd: "sao chưa thấy tiền" vs. "hoàn tiền được xử lý trong vòng X ngày"), nên
    embed thẳng query nhiều khi lệch xa embedding của đoạn văn bản trả lời đúng.
    HyDE khắc phục bằng cách: cho LLM sinh trước một đoạn văn bản giả định trả
    lời câu hỏi (viết theo văn phong tài liệu chính sách), rồi embed VÀ SEARCH
    bằng đoạn giả định đó thay vì query gốc — vì đoạn giả định này gần với không
    gian embedding của chunk thật hơn là câu hỏi thô của user.
"""

import os


def generate_hypothetical_document(query: str) -> str:
    """
    Sinh một đoạn văn bản giả định (hypothetical document) trả lời cho query,
    dùng làm input embedding cho HyDE thay vì embed thẳng query gốc.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Thiếu OPENAI_API_KEY. Hãy copy .env.example thành .env và điền API key.")

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn viết một đoạn văn bản ngắn (2-4 câu) như thể trích từ tài liệu "
                    "chính sách thương mại điện tử hoặc bài hướng dẫn hỗ trợ khách hàng "
                    "(thanh toán, đổi trả, hoàn tiền, giao hàng, quy định người bán) trả "
                    "lời trực tiếp cho câu hỏi. Không cần đúng sự thật tuyệt đối — chỉ cần "
                    "đúng văn phong và thuật ngữ của tài liệu chính sách thật."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: Nếu True, sinh một đoạn văn bản giả định (HyDE) từ query rồi
            dùng đoạn đó để embed/search thay vì embed thẳng query gốc.

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # TODO: Implement semantic search
    #
    # Bước 1: Embed query bằng cùng model ở Task 4
    # Bước 2: Query vector store (cosine similarity)
    # Bước 3: Return top_k results
    #
    # Ví dụ với ChromaDB:
    # from .task4_chunking_indexing import get_collection, get_embedding_model
    #
    # model = get_embedding_model()
    # query_vector = model.encode(query).tolist()
    # (Nếu Task 4 dùng embed_texts() dispatch theo EMBEDDING_PROVIDER thì gọi
    #  embed_texts([query])[0] ở đây thay vì get_embedding_model().encode() —
    #  để Task 5 tự động dùng đúng provider mà không cần sửa lại.)
    #
    # collection = get_collection()
    # results = collection.query(
    #     query_embeddings=[query_vector],
    #     n_results=top_k,
    #     include=["documents", "metadatas", "distances"],
    # )
    #
    # output = []
    # for doc, meta, dist in zip(
    #     results["documents"][0], results["metadatas"][0], results["distances"][0]
    # ):
    #     score = max(0.0, 1.0 - dist)  # cosine distance → similarity
    #     output.append({"content": doc, "score": round(score, 4), "metadata": meta})
    #
    # output.sort(key=lambda x: x["score"], reverse=True)
    # return output[:top_k]
    if not query.strip() or top_k <= 0:
        return []

    from .task4_chunking_indexing import embed_texts, get_collection

    collection = get_collection()
    collection_size = collection.count()
    if collection_size == 0:
        return []

    search_text = generate_hypothetical_document(query) if use_hyde else query
    query_vector = embed_texts([search_text])[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection_size),
        include=["documents", "metadatas", "distances"],
    )
    output = [
        {
            "content": document,
            "score": round(1.0 - float(distance), 4),
            "metadata": metadata,
        }
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    query = "sao chưa thấy tiền hoàn về sau khi trả hàng"

    print("-- semantic_search (query gốc) --")
    for r in semantic_search(query, top_k=5):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

    print("\n-- semantic_search (HyDE) --")
    for r in semantic_search(query, top_k=5, use_hyde=True):
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
