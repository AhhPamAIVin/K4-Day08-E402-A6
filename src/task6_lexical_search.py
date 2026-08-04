"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25_INDEX = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _load_corpus() -> None:
    global CORPUS, _BM25_INDEX
    if CORPUS:
        return

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        metadata = {
            "source": str(relative_path).replace("\\", "/"),
            "type": relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown",
        }
        CORPUS.append({"content": content, "metadata": metadata})

    if CORPUS:
        _BM25_INDEX = build_bm25_index(CORPUS)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query.strip() or top_k <= 0:
        return []

    _load_corpus()
    if not CORPUS or _BM25_INDEX is None:
        return []

    tokenized_query = _tokenize(query)
    scores = _BM25_INDEX.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score <= 0:
            break
        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx]["metadata"],
            }
        )
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # TODO: Implement BM25 index
    #
    # from rank_bm25 import BM25Okapi
    #
    # # Tokenize - có thể đơn giản split(), hoặc dùng underthesea cho tiếng Việt
    # tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    # bm25 = BM25Okapi(tokenized_corpus)
    # return bm25
    raise NotImplementedError("Implement build_bm25_index")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    # TODO: Implement lexical search
    #
    # tokenized_query = query.lower().split()
    # scores = bm25.get_scores(tokenized_query)
    #
    # # Get top_k indices
    # import numpy as np
    # top_indices = np.argsort(scores)[::-1][:top_k]
    #
    # results = []
    # for idx in top_indices:
    #     if scores[idx] > 0:
    #         results.append({
    #             "content": CORPUS[idx]["content"],
    #             "score": float(scores[idx]),
    #             "metadata": CORPUS[idx]["metadata"]
    #         })
    # return results
    raise NotImplementedError("Implement lexical_search")


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
