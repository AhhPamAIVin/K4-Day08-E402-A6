# Bài Tập Nhóm — E-commerce Support RAG Chatbot

## Thông Tin Nhóm

Nhóm gồm 6 thành viên, phát triển hệ thống RAG hỗ trợ tra cứu chính sách thương mại điện tử Shopee. Phạm Tuấn Anh phụ trách code chính, kiến trúc tổng thể và điều phối tích hợp.

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ khách hàng liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```text
Shopee Help Center / PDF / DOCX
              │
              ▼
    Task 1–3: Thu thập và chuẩn hóa
              │
              ▼
       Markdown + metadata
              │
      ┌───────┴────────┐
      ▼                ▼
Task 4–5           Task 6
OpenAI Embedding   BM25
+ ChromaDB             │
      └───────┬────────┘
              ▼
       Task 7: RRF Fusion
              │
              ▼
 Task 9: Cosine threshold check
      ┌───────┴────────┐
      ▼                ▼
Hybrid results    Task 8 PageIndex fallback
      └───────┬────────┘
              ▼
 Task 10: LLM + citation + output audit
              │
              ▼
       Streamlit Chatbot
```

### Công nghệ chính

- **Dữ liệu:** PDF, DOCX, JSON và Markdown.
- **Chunking:** `RecursiveCharacterTextSplitter`, size 800, overlap 100.
- **Embedding:** OpenAI `text-embedding-3-small`, vector 1536 chiều.
- **Vector store:** ChromaDB local persistent.
- **Lexical retrieval:** BM25.
- **Fusion:** Reciprocal Rank Fusion (RRF).
- **Fallback:** PageIndex Cloud hoặc local structured passage fallback.
- **Generation:** OpenAI/OpenRouter, câu trả lời có citation `[S1]`, `[S2]`.
- **Giao diện:** Streamlit, có tab hỏi đáp và chẩn đoán retrieval.

---

## Phân Công Công Việc

| Role | Thành viên | MSSV | Nhiệm vụ chính | Trạng thái |
|---|---|---|---|---|
| **Role 1 — Team Leader & RAG Architect** | **Phạm Tuấn Anh** | **2A202601072** | Leader; quản lý nhóm; thiết kế kiến trúc; viết và tích hợp code chính; Supervisor pipeline; điều phối demo | Hoàn thành code và tích hợp |
| **Role 2 — Data Engineering & Scraping Dev** | **Tống Duy An** | **2A202601995** | Task 1: thu thập chính sách; Task 2: crawl bài trợ giúp; Task 3: chuyển Markdown và kiểm tra metadata | Hoàn thành dữ liệu đầu vào |
| **Role 3 — Vector Database & Dense Search Dev** | **Đào Bình Minh** | **2A202601364** | Task 4: chunking, OpenAI embedding, ChromaDB indexing; Task 5: Semantic Search và HyDE | Hoàn thành chunking/indexing |
| **Role 4 — Sparse Retrieval & Fallback Dev** | **Nguyễn Việt Đăng Khoa** | **2A202601794** | Task 6: BM25; Task 7: RRF/MMR reranking; Task 8: PageIndex fallback; hỗ trợ retrieval pipeline | Hoàn thành retrieval/fallback |
| **Role 5 — Frontend UI & App Integration Dev** | **Ngô Mạnh Minh Huy** | **2A202601926** | Thiết kế Streamlit Chatbot; tích hợp Task 9–10; hiển thị citation, source và tab Analytics | Hoàn thành giao diện |
| **Role 6 — Evaluation & Benchmark QA Dev** | **Ngô Trọng Bảo** | **2A202601024** | Mở rộng `golden_dataset.json`; chạy RAGAS benchmark; A/B testing; tổng hợp `results.md` | Đang hoàn thiện benchmark |

### Phạm vi trách nhiệm theo Task

| Task | Nội dung | Role phụ trách |
|---:|---|---|
| 1–3 | Thu thập, crawl, chuẩn hóa Markdown | Role 2 — Tống Duy An |
| 4–5 | Chunking, embedding, ChromaDB, Semantic Search, HyDE | Role 3 — Đào Bình Minh |
| 6–8 | BM25, RRF/MMR, PageIndex fallback | Role 4 — Nguyễn Việt Đăng Khoa |
| 9 | Hybrid Retrieval Pipeline | Role 1 phối hợp Role 3–4 |
| 10 | Generation có citation và output audit | Role 1 phối hợp Role 5 |
| UI | Streamlit và Analytics | Role 5 — Ngô Mạnh Minh Huy |
| Evaluation | Golden dataset, RAGAS, báo cáo A/B | Role 6 — Ngô Trọng Bảo |

---

## Hướng Dẫn Chạy

```powershell
# Đứng tại thư mục dự án
cd D:\K4-Day08-E402-A6

# Kích hoạt môi trường
.\.venv\Scripts\Activate.ps1

# Cài dependency nếu chưa có
python -m pip install -r requirements.txt

# Kiểm tra tự động
python -m pytest tests\test_individual.py -v

# Chạy giao diện
streamlit run app.py
```

Truy cập `http://localhost:8501` sau khi Streamlit khởi động.

### Cấu hình môi trường

File `.env` tối thiểu:

```env
OPENAI_API_KEY=sk-proj-...
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

Nếu corpus hoặc embedding model thay đổi, chạy lại:

```powershell
python -m src.task3_convert_markdown
python -m src.task4_chunking_indexing
```

### Trạng thái hệ thống

- 13 tài liệu Markdown đã chuẩn hóa.
- 411 chunks đã được embedding và index trong ChromaDB.
- Streamlit có giao diện hỏi đáp, citation, source inspector và tab chẩn đoán.
- Automated test gần nhất: 33 passed, 2 skipped, 0 failed.
- Evaluation RAGAS là hạng mục riêng của Role 6; không ghi điểm giả khi benchmark chưa chạy.

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
