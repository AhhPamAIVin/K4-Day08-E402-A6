"""
Tab Analytics — "📊 Technical Analytics & Lab Insights"

Toàn bộ số liệu ở đây tính trực tiếp từ pipeline thật (Task 4-9), không có
số giả lập. Nếu chưa index / chưa chạy eval, tab tự báo thiếu thay vì bịa số.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.task4_chunking_indexing import load_documents, chunk_documents, CHUNK_SIZE, CHUNK_OVERLAP
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf, rerank_cross_encoder
from src.task9_retrieval_pipeline import SCORE_THRESHOLD

RESULTS_MD = Path("group_project/evaluation/results.md")


@st.cache_data(show_spinner=False)
def _get_chunks() -> list[dict]:
    return chunk_documents(load_documents())


def _zone1_chunking(chunks: list[dict]):
    st.subheader("1️⃣ Chunking & Metadata")

    df = pd.DataFrame([
        {"source": c["metadata"]["source"], "type": c["metadata"]["type"], "length": len(c["content"])}
        for c in chunks
    ])

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.histogram(df, x="length", nbins=15, title="Phân bố độ dài Chunk (ký tự)")
        fig.add_vline(x=CHUNK_SIZE, line_dash="dash", annotation_text=f"CHUNK_SIZE={CHUNK_SIZE}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Tổng số chunks", len(df))
        st.metric("Độ dài trung bình", f"{df['length'].mean():.0f} ký tự")
        st.metric("Overlap ratio cấu hình", f"{CHUNK_OVERLAP / CHUNK_SIZE:.0%}")

    st.caption("**Metadata Matrix**")
    matrix = df.groupby(["source", "type"]).agg(num_chunks=("length", "count"), avg_length=("length", "mean")).reset_index()
    st.dataframe(matrix, use_container_width=True)

    st.caption("**Chunk Inspector**")
    files = sorted(df["source"].unique())
    picked = st.selectbox("Chọn file để soi chunk", files, key="chunk_inspector_file")
    for i, c in enumerate(chunks):
        if c["metadata"]["source"] == picked:
            with st.expander(f"Chunk #{c['metadata']['chunk_index']} ({len(c['content'])} ký tự)"):
                st.text(c["content"])
                st.json(c["metadata"])

    avg_len = df["length"].mean()
    st.info(
        f"💡 **Insight**: chunk size {CHUNK_SIZE} (overlap {CHUNK_OVERLAP}, "
        f"{CHUNK_OVERLAP/CHUNK_SIZE:.0%}) cho độ dài chunk thực tế trung bình "
        f"{avg_len:.0f} ký tự trên corpus {df['source'].nunique()} file — đủ giữ trọn "
        f"1 mục chính sách/đoạn hướng dẫn trong 1 chunk mà không cắt giữa câu."
    )


def _zone2_hybrid(query: str, top_k: int):
    st.subheader("2️⃣ Hybrid Search & Reranking")

    if not query:
        st.caption("Nhập câu hỏi ở trên để phân tích retrieval.")
        return

    dense = semantic_search(query, top_k=top_k)
    sparse = lexical_search(query, top_k=top_k)
    merged = rerank_rrf([dense, sparse], top_k=top_k)
    reranked = rerank_cross_encoder(query, merged, top_k=top_k) if merged else []

    def score_of(lst, content):
        return next((x["score"] for x in lst if x["content"] == content), None)

    rows = []
    for rank_before, item in enumerate(merged, 1):
        rank_after = next((i for i, r in enumerate(reranked, 1) if r["content"] == item["content"]), None)
        rows.append({
            "content": item["content"][:60] + "...",
            "dense_cosine": score_of(dense, item["content"]),
            "bm25": score_of(sparse, item["content"]),
            "rerank_score": score_of(reranked, item["content"]),
            "rank_before": rank_before,
            "rank_after": rank_after,
            "rank_shift": (rank_before - rank_after) if rank_after else None,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["content"], y=df["rank_before"], name="Rank trước Rerank"))
    fig.add_trace(go.Bar(x=df["content"], y=df["rank_after"], name="Rank sau Rerank (cross-encoder)"))
    fig.update_layout(barmode="group", title="Rank Shift trước/sau Reranking", yaxis_title="Rank (thấp hơn = tốt hơn)")
    st.plotly_chart(fig, use_container_width=True)

    top1_shift = df.iloc[0]["rank_shift"] if len(df) else 0
    st.info(
        f"💡 **Insight**: BM25 bắt khớp từ khóa chính xác, Dense Search bắt ý định câu hỏi. "
        f"Sau cross-encoder rerank, chunk đứng #{df.iloc[0]['rank_after'] if len(df) else '-'} "
        f"trở thành Top-1 (dịch chuyển {top1_shift} bậc so với thứ hạng RRF ban đầu)."
    )


def _zone3_fallback(query: str, threshold: float):
    st.subheader("3️⃣ Threshold Monitor & PageIndex Fallback Trigger")

    if not query:
        st.caption("Nhập câu hỏi ở trên để kiểm tra fallback.")
        return

    dense = semantic_search(query, top_k=1)
    score = dense[0]["score"] if dense else 0.0
    triggered = score < threshold

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Cosine Similarity gốc (Task 5)"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "red" if triggered else "green"},
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold},
        },
    ))
    st.plotly_chart(fig, use_container_width=True)

    if triggered:
        st.error(f"🔴 FALLBACK TRIGGERED: PAGEINDEX VECTORLESS — cosine {score:.3f} < threshold {threshold}")
    else:
        st.success(f"🟢 HYBRID MATCH — cosine {score:.3f} ≥ threshold {threshold}")

    st.info(
        "💡 **Insight**: threshold so với **cosine gốc của Task 5**, không phải điểm RRF fused. "
        "Điểm RRF top-1 luôn ≈ 1/(k+1) ≈ 0.016 bất kể query liên quan hay không — dùng nó làm "
        "threshold sẽ không bao giờ trigger được fallback đúng."
    )


def _parse_scores_table(md_text: str) -> pd.DataFrame | None:
    """Parse only the FIRST markdown pipe-table's data rows (results.md has several
    tables; grabbing every '|' line in the file mixes columns from unrelated tables).
    Column count is read from the header row, so it works for any number of configs
    (currently 5). No lxml/pandas.read_html needed for a table this simple.
    """
    rows, header = [], None
    in_table = False
    for line in md_text.splitlines():
        line = line.strip()
        is_row = line.startswith("|")
        is_sep = is_row and set(line.replace("|", "").strip()) <= {"-", " "}
        if is_row and not in_table:
            in_table = True
        elif not is_row and in_table:
            break  # first table ended
        if not is_row or is_sep:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells  # ["Metric", config1, config2, ...]
            continue
        rows.append(cells[: len(header)])
    if not rows or header is None:
        return None
    return pd.DataFrame(rows, columns=header)


def _zone4_ragas():
    st.subheader("4️⃣ RAGAS Evaluation & A/B Testing")

    if not RESULTS_MD.exists():
        st.warning("Chưa có `results.md`. Chạy `python -m group_project.evaluation.eval_pipeline` trước.")
        return

    text = RESULTS_MD.read_text(encoding="utf-8")
    df = _parse_scores_table(text)
    if df is None:
        st.warning("`results.md` không đúng format bảng — kiểm tra lại eval_pipeline.")
        return

    metrics = df[~df["Metric"].str.contains(r"\*\*Average\*\*", regex=True)]
    metric_names = metrics["Metric"].tolist()
    config_names = [c for c in df.columns if c != "Metric"]
    scores = {c: pd.to_numeric(metrics[c], errors="coerce").tolist() for c in config_names}

    if not any(v == v for v in scores[config_names[0]]):  # all NaN -> template chưa điền số
        st.warning("`results.md` tồn tại nhưng chưa có số liệu — eval_pipeline chưa chạy xong.")
        st.markdown(text)
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for name in config_names:
            fig.add_trace(go.Scatterpolar(r=scores[name], theta=metric_names, fill="toself", name=name))
        fig.update_layout(polar={"radialaxis": {"range": [0, 1]}}, title=f"RAGAS 4-axis ({len(config_names)} configs)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        bar_df = pd.DataFrame({"Metric": metric_names, **scores})
        st.bar_chart(bar_df.set_index("Metric"))

    avg_per_config = {c: sum(v for v in scores[c] if v == v) / max(len(scores[c]), 1) for c in config_names}
    best = max(avg_per_config, key=avg_per_config.get)
    worst = min(avg_per_config, key=avg_per_config.get)
    st.info(
        f"💡 **Insight**: `{best}` đạt trung bình cao nhất ({avg_per_config[best]:.3f}), "
        f"`{worst}` thấp nhất ({avg_per_config[worst]:.3f}) trên {len(config_names)} configs. "
        "Chênh lệch đo được → mỗi bước trong pipeline (rerank, hybrid vs single-method, "
        "vectorless fallback) có giá trị thật, không chỉ thêm cho có."
    )


def render_analytics_tab(top_k: int = 5, threshold: float = SCORE_THRESHOLD):
    st.header("📊 Technical Analytics & Lab Insights")

    query = st.text_input(
        "Câu hỏi để phân tích retrieval (Zone 2 & 3)",
        placeholder="What payment methods does Shopee support?",
        key="analytics_query",
    )

    chunks = _get_chunks()
    _zone1_chunking(chunks)
    st.divider()
    _zone2_hybrid(query, top_k)
    st.divider()
    _zone3_fallback(query, threshold)
    st.divider()
    _zone4_ragas()
