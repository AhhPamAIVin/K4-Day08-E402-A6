# RAG Evaluation Results

## Thành viên nhóm và phân công

| Role | Thành viên | MSSV | Phụ trách |
|---|---|---|---|
| **Role 1 — Team Leader & RAG Architect** | **Phạm Tuấn Anh** | **2A202601072** | Leader; kiến trúc tổng thể; code chính; tích hợp Task 9–10; điều phối demo và hỗ trợ evaluation |
| **Role 2 — Data Engineering & Scraping Dev** | **Tống Duy An** | **2A202601995** | Task 1–3: thu thập chính sách, crawl bài trợ giúp, chuẩn hóa Markdown và metadata |
| **Role 3 — Vector Database & Dense Search Dev** | **Đào Bình Minh** | **2A202601364** | Task 4–5: chunking, OpenAI embedding, ChromaDB, Semantic Search và HyDE |
| **Role 4 — Sparse Retrieval & Fallback Dev** | **Nguyễn Việt Đăng Khoa** | **2A202601794** | Task 6–8: BM25, RRF/MMR reranking và PageIndex fallback |
| **Role 5 — Frontend UI & App Integration Dev** | **Ngô Mạnh Minh Huy** | **2A202601926** | Streamlit Chatbot, citation/source inspector, tab Analytics và tích hợp ứng dụng |
| **Role 6 — Evaluation & Benchmark QA Dev** | **Ngô Trọng Bảo** | **2A202601024** | Golden dataset 20 câu hỏi, RAGAS benchmark, A/B testing, phân tích lỗi và báo cáo kết quả |

## Phạm vi và cấu hình chạy

- Thời điểm chạy: `2026-08-04T17:04:09+07:00`
- Framework: `RAGAS 0.1.21`
- Golden dataset: **20 câu hỏi**; **40 mẫu A/B** được chấm
- Generation model: `gpt-4o-mini`; RAGAS judge: `gpt-4o-mini`
- Embedding model: `text-embedding-3-small`; `top_k=5`; temperature generation `0.0`; RAGAS `max_workers=4`
- Thời gian thực thi: **1302.2 giây**
- Lệnh: `python -m group_project.evaluation.eval_pipeline`

## Bốn chỉ số RAGAS

| Metric | Config A: Hybrid + RRF | Hợp lệ | Config B: Dense-only | Hợp lệ | Δ (A − B) |
|---|---:|---:|---:|---:|---:|
| **Faithfulness** | 0.9605 | 19/20 | 0.7899 | 19/20 | +0.1706 |
| **Answer Relevance** | 0.4289 | 17/20 | 0.3234 | 20/20 | +0.1055 |
| **Context Recall** | 1.0000 | 15/20 | 0.9766 | 19/20 | +0.0234 |
| **Context Precision** | 0.9895 | 19/20 | 1.0000 | 20/20 | -0.0105 |
| **Trung bình** | **0.8447** | — | **0.7725** | — | **+0.0722** |

## Thiết kế A/B và nhận xét

- **Config A — Hybrid + RRF:** lấy ứng viên từ dense retrieval và BM25, sau đó hợp nhất theo Reciprocal Rank Fusion.
- **Config B — Dense-only:** chỉ xếp hạng các chunk bằng cosine similarity.
- Hai nhánh dùng cùng corpus, chunking, `top_k`, prompt và generation model; do đó biến độc lập trong phép thử là chiến lược retrieval.

**Kết luận:** Config A — Hybrid + RRF đạt điểm trung bình cao hơn Config B — Dense-only **0.0722**.

Khác biệt lớn nhất nằm ở **Faithfulness**: Config A tăng **0.1706** so với Config B. Hybrid retrieval giúp câu trả lời bám bằng chứng tốt hơn rõ rệt, đồng thời tăng Answer Relevance và Context Recall. Config B nhỉnh hơn nhẹ ở Context Precision (`+0.0105`), cho thấy Dense-only có thể lấy ít context dư thừa hơn trong một số trường hợp.

## Các mẫu có điểm thấp nhất

| # | Config | Câu hỏi | Faith. | Relev. | Recall | Precision | TB | Nguồn |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | Config B — Dense-only | Người mua cần chuẩn bị bằng chứng gì khi gửi trả sản phẩm cho Shopee? | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | `legal/dang_ban_san_pham.md`, `legal/tra_hang_hoan_tien.md` |
| 2 | Config B — Dense-only | Tiền ảo như Bitcoin có được phép đăng bán trên Shopee không? | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | `legal/tra_hang_hoan_tien.md`, `legal/dang_ban_san_pham.md` |
| 3 | Config B — Dense-only | Người bán không được đăng bán những sản phẩm nào? | 0.3846 | 0.0000 | 1.0000 | 1.0000 | 0.5962 | `legal/dang_ban_san_pham.md` |
| 4 | Config B — Dense-only | Đề xuất sửa đổi Luật Hải quan đối với hàng hóa thương mại điện tử giá trị nhỏ, số lượng giao dịch lớn có… | 1.0000 | 0.0000 | N/A | 1.0000 | 0.6667 | `news/article_04.md` |
| 5 | Config A — Hybrid + RRF | Khi Người Bán tự đề xuất hoàn tiền ngay mà không cần nhận lại sản phẩm, số tiền hoàn tối thiểu là bao nh… | 0.7500 | 0.0000 | 1.0000 | 1.0000 | 0.6875 | `legal/tra_hang_hoan_tien.md`, `news/article_06.md` |

## Phân tích lỗi

### Answer Relevance là điểm yếu chính

Answer Relevance của Config A đạt `0.4289`, thấp hơn nhiều so với Faithfulness, Recall và Precision. Điều này cho thấy hệ thống thường lấy được bằng chứng đúng và bám context, nhưng câu trả lời sinh ra chưa luôn trực tiếp hoặc đầy đủ đúng trọng tâm câu hỏi.

Các nguyên nhân có thể gồm:

- context chứa nhiều điều kiện hoặc nội dung phụ;
- prompt generation chưa ép câu trả lời đi thẳng vào ý hỏi;
- câu hỏi chứa nhiều ràng buộc nhưng top-k chưa bao phủ đầy đủ;
- nguồn tự mâu thuẫn hoặc dùng cách diễn đạt khác expected answer;
- LLM trả lời thận trọng quá mức hoặc từ chối dù retrieval đã lấy đúng bằng chứng.

### Dense-only có lỗi faithfulness nghiêm trọng hơn

Hai mẫu thấp nhất của Config B có Faithfulness và Answer Relevance bằng `0`. Dense retrieval có thể tìm được tài liệu gần nghĩa nhưng không phải đoạn trả lời trực tiếp nhất. Khi thiếu BM25 và RRF, các tên riêng hoặc cụm từ chính xác như “Bitcoin”, “bằng chứng gửi trả” dễ bị xếp sau các đoạn chỉ liên quan chung.

### Context Precision cao nhưng chưa đủ bảo đảm câu trả lời tốt

Config B đạt Context Precision `1.0000` nhưng điểm trung bình vẫn thấp hơn. Context có vẻ liên quan theo judge không đồng nghĩa generation chắc chắn trả lời đúng. Chất lượng cuối cùng còn phụ thuộc việc context có chứa đúng chi tiết cần thiết và LLM có sử dụng chi tiết đó hay không.

## Khuyến nghị

1. **Ưu tiên cải thiện Answer Relevance**, vì đây là trục thấp nhất của cấu hình tham chiếu (`0.4289`). Prompt nên yêu cầu trả lời trực tiếp trong câu đầu, sau đó mới giải thích điều kiện và ngoại lệ.
2. **Chuẩn hóa granularity giữa BM25 và Dense Search.** Tại thời điểm benchmark, BM25 xếp hạng toàn văn tài liệu trong khi Dense Search xếp hạng chunk. Sau benchmark, `task6_lexical_search.py` đã được sửa để dùng cùng tập chunk của Task 4. Cần chạy lại RAGAS để đo tác động của thay đổi này.
3. **Phân tích bottom-5 theo hai tầng:** kiểm tra source/chunk retrieval trước, sau đó mới điều chỉnh prompt generation. Với lỗi retrieval, thử query expansion hoặc chunking theo Markdown header; với lỗi faithfulness, siết yêu cầu chỉ trả lời mệnh đề có citation.
4. **Bổ sung kiểm định citation và số lượng.** Code hiện đã có output audit cho citation không hợp lệ và danh sách có số đếm mâu thuẫn; nên đưa các trường hợp này vào golden dataset lần chạy tiếp theo.
5. **Calibrate lại cosine threshold** sau khi thay đổi corpus/chunking. Ngưỡng `0.46` không nên được xem là cố định cho mọi phiên bản dữ liệu.
6. **Tách câu hỏi nhiều ý định** hoặc tăng candidate pool trước RRF, tránh để `top_k=5` bỏ sót một phần bằng chứng.

## Giới hạn phép đo

- RAGAS dùng LLM-as-a-judge nên điểm có thể dao động nhẹ giữa các lần chạy, dù temperature đã đặt bằng `0`.
- `expected_answer` được dùng làm ground truth cho Context Recall/Precision; `expected_context` là nhãn kiểm tra thủ công, không phải nguyên văn context đưa cho RAGAS.
- Các giá trị `N/A` là metric bị lỗi hoặc timeout; cột **Hợp lệ** cho biết mẫu số thực sự được dùng khi tính trung bình.
- Lần chạy này thu được **148/160** điểm metric; **12** kết quả còn lại là `N/A` sau cơ chế retry.
- Log thực thi ghi nhận rate-limit TPM HTTP `429`, có thể ảnh hưởng tính đầy đủ của kết quả.
- Benchmark phản ánh phiên bản code và dữ liệu tại thời điểm `2026-08-04T17:04:09+07:00`. Những thay đổi sau đó, đặc biệt việc BM25 chuyển sang cùng đơn vị chunk, chưa được phản ánh trong bảng điểm.

## Trạng thái sau benchmark

Các thay đổi đã được thực hiện sau lần chạy trên:

- BM25 và Dense Search dùng cùng đơn vị chunk.
- ChromaDB đã index 411 chunks bằng `text-embedding-3-small`.
- Citation `[S1]`, `[S2]` đã được đồng bộ với thứ tự nguồn hiển thị.
- Thêm output audit cho citation sai và số lượng không nhất quán.
- Streamlit được bổ sung tab chẩn đoán Dense/BM25/RRF.

Vì pipeline đã thay đổi, lần benchmark tiếp theo nên được lưu thành một bảng kết quả mới để so sánh **trước và sau cải tiến**, không ghi đè số liệu lịch sử ở trên.

## Cách chạy lại benchmark

```powershell
cd D:\K4-Day08-E402-A6
.\.venv\Scripts\Activate.ps1
python -m group_project.evaluation.eval_pipeline
```

Khi chạy lại cần:

1. giữ nguyên golden dataset 20 câu để so sánh công bằng;
2. ghi timestamp và phiên bản config mới;
3. lưu số mẫu hợp lệ cho từng metric;
4. giữ lại log HTTP 429/timeout;
5. so sánh trực tiếp với baseline `0.8447` của Hybrid + RRF hiện tại.
