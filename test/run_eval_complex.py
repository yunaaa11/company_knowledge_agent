import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.retrieval.intent import classify_query_intent

DATASET_PATH = Path("test/eval_datasets/complex_policy_qa.csv")
REPORT_DIR = Path("reports/v2_complex_eval")

#清洗 CSV 中的原始数据
def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_chat_history(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def split_multi(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]

#核心业务指标计算函数
def source_hit(expected_sources, retrieved_sources, top_k=None):
    if not expected_sources:
        return False
    candidates = retrieved_sources[:top_k] if top_k else retrieved_sources
    normalized = [Path(src).name for src in candidates]
    for expected in expected_sources:
        expected_name = Path(expected).name
        if any(expected_name in src or src in expected_name for src in normalized):
            return True
    return False


def policy_hit(expected_policy_type, retrieved_policy_types):
    if not expected_policy_type or expected_policy_type == "general":
        return True
    expected = set(str(expected_policy_type).replace("/", "|").split("|"))
    expected = {item.strip() for item in expected if item.strip()}
    return bool(expected.intersection(set(retrieved_policy_types)))


def is_rejection(answer):
    markers = [
        "\u672a\u627e\u5230",
        "\u6ca1\u6709\u627e\u5230",
        "\u8d44\u6599\u4e2d\u672a",
        "\u65e0\u6cd5\u627e\u5230",
        "\u672a\u68c0\u7d22\u5230",
        "\u4e0d\u5728\u8d44\u6599",
    ]
    return any(marker in (answer or "") for marker in markers)


def base_report_row(row, final_state, latency):
    docs = final_state.get("documents", []) or []
    answer = final_state.get("answer", "") or ""
    #核心数据提取
    retrieved_sources = [doc.metadata.get("source", "unknown") for doc in docs]
    retrieved_policy_types = []
    for doc in docs:
        metadata = doc.metadata or {}
        policy_types = metadata.get("policy_types") or [metadata.get("policy_type", "unknown")]
        if isinstance(policy_types, str):
            policy_types = [policy_types]
        retrieved_policy_types.extend(policy_types)
    scores = []
    for doc in docs:
        score = doc.metadata.get("relevance_score", doc.metadata.get("weighted_score", 0.0))
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            scores.append(0.0)
    query_kinds = Counter(doc.metadata.get("query_kind", "unknown") for doc in docs)
    retrieval_query_stats = final_state.get("retrieval_query_stats", {}) or {}
    counts_by_kind = retrieval_query_stats.get("counts_by_kind", {}) or {}

    expected_sources = split_multi(row.get("expected_sources"))
    expected_policy_type = row.get("expected_policy_type", "")
    #预测的意图是否匹配预期政策类型
    intent = classify_query_intent(row["question"], chat_history=parse_chat_history(row.get("chat_history"))) or "general"

    return {
        "id": row.get("id", ""),
        "category": row.get("category", ""),
        "question": row["question"],
        "ground_truth": row.get("ground_truth", ""),
        "answer": answer,
        "intent": intent,
        "expected_policy_type": expected_policy_type,
        "intent_correct": intent == expected_policy_type or expected_policy_type == "general" or intent in str(expected_policy_type).split("/"),
        "hit_expected_policy": policy_hit(expected_policy_type, retrieved_policy_types),
        "top1_source_hit": source_hit(expected_sources, retrieved_sources, top_k=1),
        "top3_source_hit": source_hit(expected_sources, retrieved_sources, top_k=3),
        "retrieved_sources": " | ".join(retrieved_sources),
        "retrieved_policy_types": " | ".join(retrieved_policy_types),
        "retrieval_count": len(docs),
        "unique_source_count": len(set(retrieved_sources)),
        "top_relevance_score": round(scores[0], 4) if scores else 0.0,
        "avg_relevance_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "rewrite_query": " | ".join(final_state.get("rewrite_query", [])) if isinstance(final_state.get("rewrite_query"), list) else final_state.get("rewrite_query", ""),
        "hyde_query": final_state.get("hyde_query", ""),
        "query_kind_distribution": json.dumps(dict(query_kinds), ensure_ascii=False),
        "original_query_doc_count": int(counts_by_kind.get("original", 0)),
        "rewrite_query_doc_count": int(counts_by_kind.get("rewrite", 0)),
        "hyde_query_doc_count": int(counts_by_kind.get("hyde", 0)),
        "retrieval_query_stats": json.dumps(retrieval_query_stats, ensure_ascii=False),
        "retrieval_cache_hit": bool(final_state.get("retrieval_cache_hit", False)),
        "answer_cache_hit": bool(final_state.get("cache_hit", False)),
        "latency_seconds": round(latency, 3),
        "is_no_answer": parse_bool(row.get("is_no_answer", False)),
        "no_answer_rejected": is_rejection(answer) if parse_bool(row.get("is_no_answer", False)) else pd.NA,
        "difficulty": row.get("difficulty", ""),
        "answer_type": row.get("answer_type", ""),
        "cache_candidate": parse_bool(row.get("cache_candidate", False)),
    }

#汇总统计器
def build_summary(report_df):
    def numeric(col):
        if col not in report_df.columns:
            return pd.Series([0] * len(report_df))
        return pd.to_numeric(report_df[col], errors="coerce").fillna(0)

    no_answer_df = report_df[report_df["is_no_answer"] == True]
    return pd.DataFrame([{
        "sample_count": len(report_df),
        "avg_faithfulness": round(numeric("faithfulness").mean(), 4),
        "avg_answer_relevancy": round(numeric("answer_relevancy").mean(), 4),
        "avg_context_precision": round(numeric("context_precision").mean(), 4),
        "avg_context_recall": round(numeric("context_recall").mean(), 4),
        "avg_strict_score": round(numeric("strict_score").mean(), 4),
        "intent_accuracy": round(report_df["intent_correct"].mean(), 4),
        "expected_policy_hit_rate": round(report_df["hit_expected_policy"].mean(), 4),
        "top1_source_hit_rate": round(report_df["top1_source_hit"].mean(), 4),
        "top3_source_hit_rate": round(report_df["top3_source_hit"].mean(), 4),
        "no_answer_rejection_accuracy": round(no_answer_df["no_answer_rejected"].dropna().mean(), 4) if len(no_answer_df) else 0.0,
        "avg_latency_seconds": round(numeric("latency_seconds").mean(), 3),
        "avg_retrieval_count": round(numeric("retrieval_count").mean(), 2),
        "retrieval_cache_hit_rate": round(report_df["retrieval_cache_hit"].mean(), 4) if "retrieval_cache_hit" in report_df.columns else 0.0,
        "answer_cache_hit_rate": round(report_df["answer_cache_hit"].mean(), 4) if "answer_cache_hit" in report_df.columns else 0.0,
    }])


def ensure_metric_columns(report_df):
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "strict_score", "keyword_hit_ratio"]:
        if col not in report_df.columns:
            report_df[col] = pd.NA
    for col in ["eval_note", "eval_error"]:
        if col not in report_df.columns:
            report_df[col] = ""
    return report_df


def write_checkpoint(rows, skip_ragas):
    if not rows:
        return
    suffix = "retrieval_only" if skip_ragas else "ragas"
    report_df = ensure_metric_columns(pd.DataFrame(rows))
    report_df.to_csv(REPORT_DIR / f"eval_report_{suffix}_checkpoint.csv", index=False, encoding="utf-8-sig")


async def invoke_with_retry(app, inputs, retries=2):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return await app.ainvoke(inputs)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(2 * (attempt + 1))
    raise last_error


async def run_eval(skip_ragas=False, limit=None, ids=None):
    from src.agent.workflow import create_graph
    from src.llm_factory import create_chat_openai
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.reranker import RerankProcessor
    from src.retrieval.vector_store import VectorStoreManager

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(DATASET_PATH)
    if ids:
        wanted_ids = {item.strip() for item in ids.split(",") if item.strip()}
        dataset = dataset[dataset["id"].isin(wanted_ids)]
    if limit:
        dataset = dataset.head(limit)
    dataset = dataset.reset_index(drop=True)

    llm = create_chat_openai(temperature=0)
    vm = VectorStoreManager()
    hs = HybridSearcher(vm)
    reranker = RerankProcessor(hs.get_ensemble_retriever())
    app = create_graph(vm, reranker, llm)
    evaluator = None
    if not skip_ragas: # 运行时如果没加 --skip-ragas，则初始化评估器
        from src.evaluation.ragas_metrics import RagasEvaluator

        evaluator = RagasEvaluator(llm, embeddings=vm.embeddings)
   
    rows = []
    for idx, row in dataset.iterrows():
        print(f"[{idx + 1}/{len(dataset)}] {row['id']} {row['category']} - {row['question']}")
        # 从 chat_history 中提取上一轮的 intent，用于追问继承
        chat_history = parse_chat_history(row.get("chat_history"))
        prev_intent = None
        if chat_history:
            for item in reversed(chat_history):
                if isinstance(item, dict) and item.get("role") == "user":
                    prev_intent = classify_query_intent(item.get("content", ""))
                    break

        inputs = {
            "query": row["question"],
            "chat_history": chat_history,
            "loop_step": 0,
            "prev_intent": prev_intent,
            "prev_documents": [],
        }
        started = time.perf_counter()
        try:
            final_state = await invoke_with_retry(app, inputs)
            latency = time.perf_counter() - started
            report_row = base_report_row(row, final_state, latency)
        except Exception as exc:
            latency = time.perf_counter() - started
            expected_policy_type = row.get("expected_policy_type", "")
            intent = classify_query_intent(row["question"], chat_history=chat_history) or "general"
            report_row = {
                "id": row.get("id", ""),
                "category": row.get("category", ""),
                "question": row["question"],
                "ground_truth": row.get("ground_truth", ""),
                "answer": "",
                "intent": intent,
                "expected_policy_type": expected_policy_type,
                "intent_correct": intent == expected_policy_type or expected_policy_type == "general" or intent in str(expected_policy_type).split("/"),
                "hit_expected_policy": False,
                "top1_source_hit": False,
                "top3_source_hit": False,
                "retrieved_sources": "",
                "retrieved_policy_types": "",
                "retrieval_count": 0,
                "unique_source_count": 0,
                "top_relevance_score": 0.0,
                "avg_relevance_score": 0.0,
                "rewrite_query": "",
                "hyde_query": "",
                "query_kind_distribution": "{}",
                "original_query_doc_count": 0,
                "rewrite_query_doc_count": 0,
                "hyde_query_doc_count": 0,
                "retrieval_query_stats": "{}",
                "retrieval_cache_hit": False,
                "answer_cache_hit": False,
                "latency_seconds": round(latency, 3),
                "is_no_answer": parse_bool(row.get("is_no_answer", False)),
                "no_answer_rejected": pd.NA,
                "difficulty": row.get("difficulty", ""),
                "answer_type": row.get("answer_type", ""),
                "cache_candidate": parse_bool(row.get("cache_candidate", False)),
                "eval_error": f"agent_error: {exc}",
            }
            rows.append(report_row)
            write_checkpoint(rows, skip_ragas)
            continue
        #RAGAS 评分环节
        if evaluator:  # 运行时如果加了--skip-ragas参数，evaluator 为 None，这里不执行
            try:
                if parse_bool(row.get("is_no_answer", False)):
                    report_row["faithfulness"] = pd.NA
                    report_row["answer_relevancy"] = pd.NA
                    report_row["context_precision"] = pd.NA
                    report_row["context_recall"] = pd.NA
                    report_row["keyword_hit_ratio"] = pd.NA
                    report_row["strict_score"] = 1.0 if bool(report_row.get("no_answer_rejected")) else 0.0
                    report_row["eval_note"] = "skip_ragas_for_no_answer"
                    raise StopIteration
                score_df = evaluator.evaluate_response(
                    row["question"],
                    {"answer": final_state.get("answer", ""), "documents": final_state.get("documents", [])},
                    row.get("ground_truth", ""),
                )
                for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "strict_score", "keyword_hit_ratio"]:
                    if col in score_df.columns:
                        report_row[col] = score_df.iloc[0][col]
            except Exception as exc:
                if not isinstance(exc, StopIteration):
                    report_row["eval_error"] = str(exc)
        rows.append(report_row)
        write_checkpoint(rows, skip_ragas)

    report_df = ensure_metric_columns(pd.DataFrame(rows))

    summary_df = build_summary(report_df)
    category_summary = report_df.groupby("category", dropna=False).agg(
        sample_count=("id", "count"),
        avg_strict_score=("strict_score", lambda s: round(pd.to_numeric(s, errors="coerce").fillna(0).mean(), 4)),
        expected_policy_hit_rate=("hit_expected_policy", "mean"),
        top3_source_hit_rate=("top3_source_hit", "mean"),
        intent_accuracy=("intent_correct", "mean"),
        retrieval_cache_hit_rate=("retrieval_cache_hit", "mean"),
        avg_latency_seconds=("latency_seconds", "mean"),
    ).reset_index()
    #检索质量最差的样本单独拎出来供人工分析。
    category_summary["expected_policy_hit_rate"] = category_summary["expected_policy_hit_rate"].round(4)
    category_summary["top3_source_hit_rate"] = category_summary["top3_source_hit_rate"].round(4)
    category_summary["intent_accuracy"] = category_summary["intent_accuracy"].round(4)
    category_summary["retrieval_cache_hit_rate"] = category_summary["retrieval_cache_hit_rate"].round(4)
    category_summary["avg_latency_seconds"] = category_summary["avg_latency_seconds"].round(3)

    error_cases = report_df[(report_df["hit_expected_policy"] == False) | (report_df["top3_source_hit"] == False)]
    #全量报告、汇总、分类汇总、错误案例、测试集副本
    suffix = "retrieval_only" if skip_ragas else "ragas"
    report_df.to_csv(REPORT_DIR / f"eval_report_{suffix}.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(REPORT_DIR / f"eval_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    category_summary.to_csv(REPORT_DIR / f"category_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    error_cases.to_csv(REPORT_DIR / f"error_cases_{suffix}.csv", index=False, encoding="utf-8-sig")
    dataset.to_csv(REPORT_DIR / f"testset_complex_{suffix}.csv", index=False, encoding="utf-8-sig")

    report_df.to_csv(REPORT_DIR / "eval_report.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(REPORT_DIR / "eval_summary.csv", index=False, encoding="utf-8-sig")
    category_summary.to_csv(REPORT_DIR / "category_summary.csv", index=False, encoding="utf-8-sig")
    error_cases.to_csv(REPORT_DIR / "error_cases.csv", index=False, encoding="utf-8-sig")
    dataset.to_csv(REPORT_DIR / "testset_complex.csv", index=False, encoding="utf-8-sig")

    print("\nSaved reports to", REPORT_DIR)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ragas", action="store_true", help="Only run retrieval diagnostics and skip RAGAS scoring.")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N samples for smoke tests.")
    parser.add_argument("--ids", default=None, help="Comma-separated sample ids to run, for example Q029,Q030.")
    args = parser.parse_args()
    asyncio.run(run_eval(skip_ragas=args.skip_ragas, limit=args.limit, ids=args.ids))
