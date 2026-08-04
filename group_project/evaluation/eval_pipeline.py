"""Run a reproducible RAGAS A/B evaluation for the project RAG pipeline.

The benchmark compares the current hybrid retriever (dense + BM25 + RRF) with
dense-only retrieval.  Both configurations share the same chunks, top-k,
generation model, prompt, and RAGAS judge so that retrieval is the only changed
variable.

Run from the repository root:

    python -m group_project.evaluation.eval_pipeline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


EVALUATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALUATION_DIR.parents[1]
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"
EVAL_CACHE_DIR = REPO_ROOT / ".venv" / "evaluation_cache"
EVAL_CHROMA_DIR = EVAL_CACHE_DIR / "chroma"
GENERATION_CACHE_PATH = EVAL_CACHE_DIR / "generated_answers.json"

CONFIG_LABELS = {
    "hybrid_rrf": "Config A — Hybrid + RRF",
    "dense_only": "Config B — Dense-only",
}
METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevance",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}


def load_golden_dataset() -> list[dict[str, str]]:
    """Load and validate the golden dataset."""
    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("golden_dataset.json must contain a JSON array")

    required = ("question", "expected_answer", "expected_context")
    for index, item in enumerate(dataset, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Test case {index} must be a JSON object")
        missing = [field for field in required if not str(item.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Test case {index} is missing: {', '.join(missing)}")
    return dataset


def _safe_float(value: Any) -> float | None:
    """Convert a metric value to a finite float, otherwise return None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:40]


class EvaluationPipeline:
    """Evaluation adapter around the project's chunking/retrieval/generation code."""

    def __init__(self, top_k: int, generation_model: str):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.generation_model = generation_model
        self._answer_cache = self._load_answer_cache()

    @staticmethod
    def _load_answer_cache() -> dict[str, str]:
        if not GENERATION_CACHE_PATH.exists():
            return {}
        try:
            value = json.loads(GENERATION_CACHE_PATH.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_answer_cache(self) -> None:
        EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        GENERATION_CACHE_PATH.write_text(
            json.dumps(self._answer_cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _corpus_fingerprint(chunks: list[dict]) -> str:
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk["content"].encode("utf-8"))
            digest.update(
                json.dumps(chunk["metadata"], sort_keys=True, ensure_ascii=False).encode(
                    "utf-8"
                )
            )
        return digest.hexdigest()

    def _dense_retrieve_many(self, questions: list[str]) -> list[list[dict]]:
        """Build/use an isolated, dimension-safe evaluation index and query it."""
        import chromadb

        from src.task4_chunking_indexing import (
            EMBEDDING_MODEL,
            chunk_documents,
            embed_texts,
            load_documents,
        )

        chunks = chunk_documents(load_documents())
        if not chunks:
            raise RuntimeError("No standardized document chunks were found")

        fingerprint = self._corpus_fingerprint(chunks)
        collection_name = (
            f"rag-eval-{_slug(EMBEDDING_MODEL)}-{fingerprint[:12]}"
        )
        EVAL_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(EVAL_CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": EMBEDDING_MODEL,
                "corpus_fingerprint": fingerprint,
            },
        )

        if collection.count() != len(chunks):
            print(
                f"Building evaluation index: {len(chunks)} chunks "
                f"with {EMBEDDING_MODEL}...",
                flush=True,
            )
            embeddings = embed_texts([chunk["content"] for chunk in chunks])
            collection.upsert(
                ids=[f"eval-chunk-{index:04d}" for index in range(len(chunks))],
                documents=[chunk["content"] for chunk in chunks],
                embeddings=embeddings,
                metadatas=[chunk["metadata"] for chunk in chunks],
            )

        query_embeddings = embed_texts(questions)
        n_results = min(self.top_k * 2, collection.count())
        raw = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        batches: list[list[dict]] = []
        for documents, metadatas, distances in zip(
            raw["documents"], raw["metadatas"], raw["distances"]
        ):
            results = [
                {
                    "content": document,
                    "score": round(1.0 - float(distance), 6),
                    "metadata": metadata,
                    "source": "dense",
                }
                for document, metadata, distance in zip(
                    documents, metadatas, distances
                )
            ]
            batches.append(results)
        return batches

    def retrieve_configs(self, questions: list[str]) -> dict[str, list[list[dict]]]:
        """Retrieve contexts for both A/B configs using shared dense queries."""
        from src.task6_lexical_search import lexical_search
        from src.task7_reranking import rerank_rrf

        dense_batches = self._dense_retrieve_many(questions)
        dense_only: list[list[dict]] = []
        hybrid_rrf: list[list[dict]] = []

        for question, dense_results in zip(questions, dense_batches):
            dense_only.append([dict(item) for item in dense_results[: self.top_k]])
            sparse_results = lexical_search(question, top_k=self.top_k * 2)
            hybrid = rerank_rrf(
                [dense_results, sparse_results],
                top_k=self.top_k,
            )
            for item in hybrid:
                item["source"] = "hybrid"
            hybrid_rrf.append(hybrid)

        return {"hybrid_rrf": hybrid_rrf, "dense_only": dense_only}

    def generate_with_citation(
        self,
        question: str,
        chunks: list[dict],
        config_name: str,
    ) -> str:
        """Generate one grounded answer, reusing Task 10's prompt utilities."""
        from openai import OpenAI

        from src.task10_generation import SYSTEM_PROMPT, format_context, reorder_for_llm

        if not chunks:
            return "Tôi không thể xác minh thông tin này từ nguồn hiện có"

        context = format_context(reorder_for_llm(chunks))
        cache_payload = {
            "question": question,
            "config": config_name,
            "model": self.generation_model,
            "system_prompt": SYSTEM_PROMPT,
            "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cached = self._answer_cache.get(cache_key)
        if cached:
            return cached

        client = OpenAI(timeout=180.0, max_retries=4)
        response = client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\n---\n\nQuestion: {question}",
                },
            ],
            temperature=0.0,
            max_tokens=700,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("Generation model returned an empty answer")
        self._answer_cache[cache_key] = answer
        self._save_answer_cache()
        return answer

    def collect_samples(
        self, golden_dataset: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Run retrieval and generation for every question under both configs."""
        questions = [item["question"] for item in golden_dataset]
        retrieved = self.retrieve_configs(questions)
        total = len(golden_dataset) * len(CONFIG_LABELS)
        completed = 0
        samples: list[dict[str, Any]] = []

        for config_name in CONFIG_LABELS:
            for sample_id, (item, chunks) in enumerate(
                zip(golden_dataset, retrieved[config_name]), 1
            ):
                completed += 1
                print(
                    f"[{completed:02d}/{total}] Generating "
                    f"{config_name} / case {sample_id:02d}",
                    flush=True,
                )
                answer = self.generate_with_citation(
                    item["question"], chunks, config_name
                )
                samples.append(
                    {
                        "sample_id": sample_id,
                        "config": config_name,
                        "question": item["question"],
                        "answer": answer,
                        "contexts": [chunk["content"] for chunk in chunks],
                        "ground_truth": item["expected_answer"],
                        "expected_context": item["expected_context"],
                        "retrieval_sources": ", ".join(
                            sorted({chunk.get("source", "unknown") for chunk in chunks})
                        ),
                        "source_files": ", ".join(
                            dict.fromkeys(
                                str(chunk.get("metadata", {}).get("source", "unknown"))
                                for chunk in chunks
                            )
                        ),
                    }
                )
        return samples


def evaluate_with_ragas(
    generated_samples: list[dict[str, Any]],
    judge_model: str,
    embedding_model: str,
    max_workers: int,
) -> dict[str, Any]:
    """Score generated samples with the four required RAGAS metrics."""
    import ragas
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    dataset = Dataset.from_list(generated_samples)
    judge = ChatOpenAI(
        model=judge_model,
        temperature=0.0,
        timeout=180,
        max_retries=4,
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        timeout=180,
        max_retries=4,
    )
    run_config = RunConfig(
        timeout=180,
        max_retries=4,
        max_wait=30,
        max_workers=max_workers,
        seed=42,
    )

    print(
        f"Scoring {len(generated_samples)} samples with RAGAS "
        f"({judge_model}, workers={max_workers})...",
        flush=True,
    )
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    scored_rows: list[dict[str, Any]] = []
    for record in result.to_pandas().to_dict(orient="records"):
        for metric in METRICS:
            record[metric] = _safe_float(record.get(metric))
        scored_rows.append(record)
    return {
        "framework": "RAGAS",
        "framework_version": ragas.__version__,
        "rows": scored_rows,
    }


def compare_configs(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metric means, valid counts, deltas, and worst performers."""
    summary: dict[str, dict[str, Any]] = {}
    for config_name in CONFIG_LABELS:
        rows = [row for row in scored_rows if row["config"] == config_name]
        metric_summary: dict[str, Any] = {}
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            metric_summary[metric] = {
                "mean": sum(values) / len(values) if values else None,
                "valid": len(values),
                "total": len(rows),
            }
        valid_means = [
            metric_summary[metric]["mean"]
            for metric in METRICS
            if metric_summary[metric]["mean"] is not None
        ]
        summary[config_name] = {
            "metrics": metric_summary,
            "average": sum(valid_means) / len(valid_means) if valid_means else None,
            "rows": len(rows),
            "retrieval_sources": dict(
                Counter(str(row.get("retrieval_sources", "unknown")) for row in rows)
            ),
        }

    deltas: dict[str, float | None] = {}
    for metric in METRICS:
        score_a = summary["hybrid_rrf"]["metrics"][metric]["mean"]
        score_b = summary["dense_only"]["metrics"][metric]["mean"]
        deltas[metric] = (
            score_a - score_b
            if score_a is not None and score_b is not None
            else None
        )

    ranked_rows = []
    for row in scored_rows:
        values = [row[metric] for metric in METRICS if row[metric] is not None]
        row_copy = dict(row)
        row_copy["metric_average"] = (
            sum(values) / len(values) if values else None
        )
        ranked_rows.append(row_copy)
    worst = sorted(
        ranked_rows,
        key=lambda row: (
            row["metric_average"] is not None,
            row["metric_average"] if row["metric_average"] is not None else -1,
        ),
    )[:5]

    return {"summary": summary, "deltas": deltas, "worst": worst}


def _format_score(value: Any) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:.4f}"


def _format_delta(value: Any) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:+.4f}"


def _markdown_cell(value: Any, max_length: int | None = None) -> str:
    text = str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    if max_length and len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "…"
    return text


def export_results(
    results: dict[str, Any],
    comparison: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Render the measured scores and diagnostics to results.md."""
    summary = comparison["summary"]
    score_a = summary["hybrid_rrf"]["average"]
    score_b = summary["dense_only"]["average"]
    if score_a is None or score_b is None:
        conclusion = "Không thể xác định cấu hình tốt hơn vì thiếu điểm hợp lệ."
        winner = None
    elif abs(score_a - score_b) < 1e-12:
        conclusion = "Hai cấu hình có điểm trung bình bốn chỉ số bằng nhau."
        winner = None
    else:
        winner = "hybrid_rrf" if score_a > score_b else "dense_only"
        loser = "dense_only" if winner == "hybrid_rrf" else "hybrid_rrf"
        gap = abs(score_a - score_b)
        conclusion = (
            f"{CONFIG_LABELS[winner]} đạt điểm trung bình cao hơn "
            f"{CONFIG_LABELS[loser]} **{gap:.4f}**."
        )

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Phạm vi và cấu hình chạy",
        "",
        f"- Thời điểm chạy: `{metadata['finished_at']}`",
        f"- Framework: `{results['framework']} {results['framework_version']}`",
        f"- Golden dataset: **{metadata['question_count']} câu hỏi**; "
        f"**{metadata['sample_count']} mẫu A/B** được chấm",
        f"- Generation model: `{metadata['generation_model']}`; "
        f"RAGAS judge: `{metadata['judge_model']}`",
        f"- Embedding model: `{metadata['embedding_model']}`; "
        f"`top_k={metadata['top_k']}`; temperature generation `0.0`; "
        f"RAGAS `max_workers={metadata['max_workers']}`",
        f"- Thời gian thực thi: **{metadata['duration_seconds']:.1f} giây**",
        "- Lệnh: `python -m group_project.evaluation.eval_pipeline`",
        "",
        "## Bốn chỉ số RAGAS",
        "",
        "| Metric | Config A: Hybrid + RRF | Hợp lệ | Config B: Dense-only | Hợp lệ | Δ (A − B) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        metric_a = summary["hybrid_rrf"]["metrics"][metric]
        metric_b = summary["dense_only"]["metrics"][metric]
        lines.append(
            f"| **{METRIC_LABELS[metric]}** | {_format_score(metric_a['mean'])} | "
            f"{metric_a['valid']}/{metric_a['total']} | {_format_score(metric_b['mean'])} | "
            f"{metric_b['valid']}/{metric_b['total']} | "
            f"{_format_delta(comparison['deltas'][metric])} |"
        )
    lines.extend(
        [
            f"| **Trung bình** | **{_format_score(score_a)}** | — | "
            f"**{_format_score(score_b)}** | — | **{_format_delta(score_a - score_b if score_a is not None and score_b is not None else None)}** |",
            "",
            "## Thiết kế A/B và nhận xét",
            "",
            "- **Config A — Hybrid + RRF:** lấy ứng viên từ dense retrieval và BM25, "
            "sau đó hợp nhất theo Reciprocal Rank Fusion.",
            "- **Config B — Dense-only:** chỉ xếp hạng các chunk bằng cosine similarity.",
            "- Hai nhánh dùng cùng corpus, chunking, `top_k`, prompt và generation model; "
            "do đó biến độc lập trong phép thử là chiến lược retrieval.",
            "",
            f"**Kết luận:** {conclusion}",
            "",
        ]
    )

    valid_deltas = {
        metric: delta
        for metric, delta in comparison["deltas"].items()
        if delta is not None
    }
    if valid_deltas:
        strongest = max(valid_deltas, key=lambda metric: abs(valid_deltas[metric]))
        direction = "tăng" if valid_deltas[strongest] >= 0 else "giảm"
        lines.extend(
            [
                f"Khác biệt lớn nhất nằm ở **{METRIC_LABELS[strongest]}**: "
                f"Config A {direction} **{abs(valid_deltas[strongest]):.4f}** so với Config B.",
                "",
            ]
        )

    lines.extend(
        [
            "## Các mẫu có điểm thấp nhất",
            "",
            "| # | Config | Câu hỏi | Faith. | Relev. | Recall | Precision | TB | Nguồn |",
            "|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(comparison["worst"], 1):
        lines.append(
            f"| {rank} | {_markdown_cell(CONFIG_LABELS[row['config']])} | "
            f"{_markdown_cell(row['question'], 105)} | "
            f"{_format_score(row['faithfulness'])} | "
            f"{_format_score(row['answer_relevancy'])} | "
            f"{_format_score(row['context_recall'])} | "
            f"{_format_score(row['context_precision'])} | "
            f"{_format_score(row['metric_average'])} | "
            f"{_markdown_cell(row.get('source_files', ''), 70)} |"
        )

    winning_summary = summary[winner] if winner else summary["hybrid_rrf"]
    lowest_metric = min(
        METRICS,
        key=lambda metric: (
            winning_summary["metrics"][metric]["mean"]
            if winning_summary["metrics"][metric]["mean"] is not None
            else float("inf")
        ),
    )
    lines.extend(
        [
            "",
            "## Khuyến nghị",
            "",
            f"1. Ưu tiên cải thiện **{METRIC_LABELS[lowest_metric]}**, vì đây là trục "
            f"thấp nhất của cấu hình tham chiếu ({_format_score(winning_summary['metrics'][lowest_metric]['mean'])}).",
            "2. BM25 và dense retrieval hiện đã xếp hạng cùng đơn vị chunk. Bước tiếp theo là "
            "hiệu chỉnh tham số RRF hoặc thử weighted fusion trên golden dataset để cân bằng "
            "tín hiệu từ khóa và cosine similarity.",
            "3. Phân tích các câu trong bảng bottom-5 theo hai tầng: kiểm tra source/chunk "
            "được lấy trước, rồi mới điều chỉnh prompt generation. Với lỗi retrieval, thử query "
            "expansion hoặc chunking theo Markdown header; với lỗi faithfulness, siết yêu cầu chỉ "
            "trả lời các mệnh đề có citation.",
            "",
            "## Giới hạn phép đo",
            "",
            "- RAGAS dùng LLM-as-a-judge nên điểm có thể dao động nhẹ giữa các lần chạy, dù "
            "temperature đã đặt bằng 0.",
            "- `expected_answer` được dùng làm ground truth cho Context Recall/Precision; "
            "`expected_context` là nhãn kiểm tra thủ công, không phải nguyên văn context đưa cho RAGAS.",
            "- Các giá trị `N/A` (nếu có) là metric bị lỗi/timeout; cột **Hợp lệ** cho biết "
            "mẫu số thực sự được dùng khi tính trung bình.",
            "",
        ]
    )

    valid_metric_count = sum(
        summary[config_name]["metrics"][metric]["valid"]
        for config_name in CONFIG_LABELS
        for metric in METRICS
    )
    total_metric_count = metadata["sample_count"] * len(METRICS)
    missing_metric_count = total_metric_count - valid_metric_count
    lines.insert(
        -1,
        f"- Lần chạy này thu được **{valid_metric_count}/{total_metric_count}** điểm metric; "
        f"**{missing_metric_count}** kết quả còn lại là `N/A` sau cơ chế retry.",
    )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (debug/rate-limit fallback)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--generation-model",
        default=os.getenv("EVAL_GENERATION_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for generation and RAGAS")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")

    golden_dataset = load_golden_dataset()
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        golden_dataset = golden_dataset[: args.limit]
    if len(golden_dataset) < 15:
        print(
            f"Warning: evaluating {len(golden_dataset)} questions; the assignment "
            "requires 15–20 for the final run.",
            flush=True,
        )

    started = time.perf_counter()
    started_at = datetime.now().astimezone()
    print(f"Loaded {len(golden_dataset)} golden test cases", flush=True)
    pipeline = EvaluationPipeline(
        top_k=args.top_k,
        generation_model=args.generation_model,
    )
    generated_samples = pipeline.collect_samples(golden_dataset)
    results = evaluate_with_ragas(
        generated_samples,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        max_workers=args.max_workers,
    )
    comparison = compare_configs(results["rows"])
    finished_at = datetime.now().astimezone()
    metadata = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": time.perf_counter() - started,
        "question_count": len(golden_dataset),
        "sample_count": len(generated_samples),
        "generation_model": args.generation_model,
        "judge_model": args.judge_model,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "max_workers": args.max_workers,
    }
    export_results(results, comparison, metadata)
    print(f"Wrote {RESULTS_PATH.relative_to(REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()