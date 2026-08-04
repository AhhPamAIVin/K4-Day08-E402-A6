"""Lightweight diagnostics tab for the Streamlit application."""

import pandas as pd
import streamlit as st

from src.task4_chunking_indexing import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents, load_documents
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task9_retrieval_pipeline import SCORE_THRESHOLD


@st.cache_data(show_spinner=False)
def corpus_summary() -> tuple[list[dict], pd.DataFrame]:
    chunks = chunk_documents(load_documents())
    rows = [{
        "source": chunk["metadata"].get("source", "unknown"),
        "type": chunk["metadata"].get("type", "unknown"),
        "customer_role": chunk["metadata"].get("customer_role", "both"),
        "length": len(chunk["content"]),
    } for chunk in chunks]
    return chunks, pd.DataFrame(rows)


def render_analytics_tab(top_k: int = 5) -> None:
    st.header("📊 Chẩn đoán retrieval")
    st.caption("Số liệu lấy trực tiếp từ corpus và pipeline, không dùng dữ liệu minh họa giả.")

    chunks, frame = corpus_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tài liệu", frame["source"].nunique() if not frame.empty else 0)
    col2.metric("Chunks", len(chunks))
    col3.metric("Độ dài TB", f"{frame['length'].mean():.0f}" if not frame.empty else "0")
    col4.metric("Overlap", f"{CHUNK_OVERLAP / CHUNK_SIZE:.0%}")

    with st.expander("Phân bố corpus", expanded=False):
        if not frame.empty:
            st.bar_chart(frame.groupby("type").size().rename("chunks"))
            st.dataframe(
                frame.groupby(["source", "type", "customer_role"])
                .agg(chunks=("length", "count"), avg_length=("length", "mean"))
                .reset_index(),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("So sánh Dense, BM25 và RRF")
    query = st.text_input(
        "Câu hỏi chẩn đoán",
        value="Shopee hỗ trợ những phương thức thanh toán nào?",
        key="diagnostic_query",
    )
    if st.button("Chạy chẩn đoán retrieval", type="primary"):
        if not query.strip():
            st.warning("Hãy nhập câu hỏi.")
            return
        try:
            with st.spinner("Đang chạy Dense Search và BM25…"):
                dense = semantic_search(query, top_k=top_k)
                sparse = lexical_search(query, top_k=top_k)
                fused = rerank_rrf([dense, sparse], top_k=top_k)

            best_dense = dense[0]["score"] if dense else 0.0
            a, b, c = st.columns(3)
            a.metric("Best cosine", f"{best_dense:.4f}")
            b.metric("Ngưỡng fallback", f"{SCORE_THRESHOLD:.2f}")
            c.metric("Quyết định", "Hybrid" if best_dense >= SCORE_THRESHOLD else "Fallback")

            rows = []
            for rank, item in enumerate(fused, 1):
                content = item["content"]
                dense_item = next((x for x in dense if x["content"] == content), None)
                sparse_item = next((x for x in sparse if x["content"] == content), None)
                rows.append({
                    "rank": rank,
                    "source": item.get("metadata", {}).get("source", "unknown"),
                    "dense_cosine": dense_item["score"] if dense_item else None,
                    "bm25": sparse_item["score"] if sparse_item else None,
                    "rrf": item["score"],
                    "preview": content[:150],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.info(
                "Fallback được quyết định bằng cosine gốc, không dùng điểm RRF. "
                "RRF chỉ dùng thứ hạng nên điểm của nó nhỏ và không biểu diễn độ liên quan tuyệt đối."
            )
        except Exception as exc:
            st.error(f"Không thể chạy chẩn đoán: {type(exc).__name__}: {exc}")
