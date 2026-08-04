"""Streamlit UI for the e-commerce support RAG system."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

st.set_page_config(
    page_title="Trợ lý chính sách TMĐT",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #f8fafc 0%, #ffffff 34%);}
    .hero {padding: 1.3rem 1.5rem; border-radius: 20px; color: white;
      background: linear-gradient(120deg, #ee4d2d, #ff7043 55%, #ff9f43);
      box-shadow: 0 12px 30px rgba(238,77,45,.18); margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .45rem 0 0; opacity: .92;}
    .status-card {border: 1px solid #e2e8f0; border-radius: 14px; padding: .8rem 1rem;
      background: rgba(255,255,255,.82);}
    [data-testid="stChatMessage"] {border: 1px solid #edf2f7; border-radius: 16px; padding: .35rem;}
    [data-testid="stSidebar"] {background: #fffaf7;}
    .source-badge {display:inline-block; padding:.15rem .5rem; border-radius:999px;
      background:#fff1ec; color:#c24124; font-size:.78rem; margin-right:.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def system_status() -> dict:
    count = 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection("ecommerce_support_docs")
        count = collection.count()
    except Exception:
        pass
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "pageindex": bool(os.getenv("PAGEINDEX_API_KEY")),
        "chunks": count,
    }


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Xem {len(sources)} đoạn nguồn đã sử dụng"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            name = metadata.get("source", "Không rõ nguồn")
            doc_type = metadata.get("type", "unknown")
            retrieval = source.get("source", "hybrid")
            score = float(source.get("score", 0.0))
            st.markdown(
                f"<span class='source-badge'>S{index}</span> **{name}**  "
                f"`{doc_type}` · `{retrieval}` · score `{score:.4f}`",
                unsafe_allow_html=True,
            )
            preview = source.get("content", "").strip()
            st.caption(preview[:650] + ("…" if len(preview) > 650 else ""))
            if index < len(sources):
                st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

status = system_status()

with st.sidebar:
    st.title("🛍️ RAG Policy Assistant")
    st.caption("Trợ lý tra cứu chính sách Shopee từ nguồn đã thu thập và lập chỉ mục.")
    st.divider()

    st.subheader("Trạng thái")
    st.markdown(f"{'🟢' if status['openai'] else '🔴'} OpenAI API")
    st.markdown(f"{'🟢' if status['chunks'] else '🔴'} ChromaDB: **{status['chunks']} chunks**")
    st.markdown(f"{'🟢' if status['openrouter'] else '⚪'} OpenRouter (tùy chọn)")
    st.markdown(f"{'🟢' if status['pageindex'] else '⚪'} PageIndex Cloud (tùy chọn)")

    st.divider()
    top_k = st.slider(
        "Số đoạn làm bằng chứng",
        min_value=3,
        max_value=8,
        value=5,
        help="Nhiều đoạn hơn tăng độ bao phủ nhưng làm context dài hơn.",
    )
    show_debug = st.toggle("Hiện chi tiết kỹ thuật", value=False)

    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    st.caption("Dense OpenAI + BM25 → RRF → fallback PageIndex → LLM có trích dẫn")

tab_chat, tab_analytics, tab_about = st.tabs(["💬 Hỏi đáp", "📊 Chẩn đoán", "ℹ️ Giới thiệu"])

with tab_chat:
    st.markdown(
        """
        <div class="hero">
          <h1>Trợ lý chính sách thương mại điện tử</h1>
          <p>Tra cứu đổi trả, hoàn tiền, thanh toán, bảo mật và quy định người bán với nguồn dẫn rõ ràng.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "Thời hạn yêu cầu trả hàng và hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Người mua cần cung cấp bằng chứng gì khi yêu cầu hoàn tiền?",
        "Những sản phẩm nào bị cấm hoặc hạn chế đăng bán?",
    ]
    cols = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        if cols[index % 2].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_query = suggestion
            st.rerun()

    if not status["openai"] or not status["chunks"]:
        missing = []
        if not status["openai"]:
            missing.append("OPENAI_API_KEY")
        if not status["chunks"]:
            missing.append("Chroma index")
        st.error("Hệ thống chưa sẵn sàng: " + ", ".join(missing))

    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "🛍️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))
            if show_debug and message.get("debug"):
                st.json(message["debug"])

    typed_query = st.chat_input("Nhập câu hỏi về chính sách Shopee…")
    query = typed_query or st.session_state.pending_query
    if query:
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)

        answer = ""
        sources = []
        debug = {}
        with st.chat_message("assistant", avatar="🛍️"):
            try:
                with st.spinner("Đang tìm bằng chứng và tổng hợp câu trả lời…"):
                    from src.task10_generation import generate_with_citation

                    result = generate_with_citation(query, top_k=top_k)
                answer = result["answer"]
                sources = result.get("sources", [])
                debug = {
                    "retrieval_source": result.get("retrieval_source", "unknown"),
                    "chunks_used": len(sources),
                    "top_k": top_k,
                }
                st.markdown(answer)
                render_sources(sources)
                if show_debug:
                    st.json(debug)
            except Exception as exc:
                answer = (
                    "Hệ thống chưa thể xử lý câu hỏi lúc này. Hãy kiểm tra API key, "
                    "kết nối mạng và Chroma index rồi thử lại."
                )
                st.error(answer)
                with st.expander("Chi tiết lỗi dành cho kỹ thuật"):
                    st.code(f"{type(exc).__name__}: {exc}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "debug": debug,
        })

with tab_analytics:
    from analytics_tab import render_analytics_tab

    render_analytics_tab(top_k=top_k)

with tab_about:
    st.header("Hệ thống này làm gì?")
    st.markdown(
        """
        Hệ thống chỉ trả lời dựa trên tài liệu trong kho dữ liệu. Mỗi câu trả lời được tạo
        sau khi kết hợp tìm kiếm ngữ nghĩa và tìm kiếm từ khóa, sau đó gắn nguồn `[S1]`,
        `[S2]` để người dùng kiểm tra lại.

        - **Phù hợp:** trả hàng, hoàn tiền, thanh toán, tài khoản, bảo mật, đăng bán.
        - **Không phù hợp:** tư vấn pháp lý cá nhân, trạng thái đơn hàng thời gian thực,
          thông tin ngoài corpus hoặc chính sách vừa thay đổi nhưng chưa được cập nhật.
        - **Nguyên tắc:** khi bằng chứng không đủ, trợ lý phải nói không thể xác minh.
        """
    )
