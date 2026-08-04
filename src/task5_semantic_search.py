"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

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

    query_vector = embed_texts([query])[0]
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
    # Test
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
