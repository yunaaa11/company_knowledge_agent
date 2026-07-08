import argparse
import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from pathlib import Path

import pandas as pd

from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import RerankProcessor



from config import Config
from run_eval_complex import base_report_row, build_summary, parse_chat_history

DATASET_PATH = Path("test/eval_datasets/complex_policy_qa.csv")
REPORT_DIR = Path("reports/v2_complex_eval")

#全局配置操控函数
def set_feature_flags(intent_filter=True, hyde=True):
    old = {
        "ENABLE_INTENT_FILTER": Config.ENABLE_INTENT_FILTER,
        "ENABLE_HYDE": Config.ENABLE_HYDE,
    }
    Config.ENABLE_INTENT_FILTER = intent_filter
    Config.ENABLE_HYDE = hyde
    return old


def restore_feature_flags(old):
    for key, value in old.items():
        setattr(Config, key, value)


def build_retriever(vm, variant):
    if variant == "baseline_vector_only":
        return vm.get_parent_retriever(), False, False

    hs = HybridSearcher(vm)
    base = hs.get_ensemble_retriever()
    if variant == "hybrid_bm25_vector":
        return base, False, False
    if variant in {"plus_rerank", "plus_intent_filter", "full_pipeline"}:
        return RerankProcessor(base), True, variant != "plus_rerank"
    return RerankProcessor(base), True, True

# 单变体运行器 
async def run_variant(name, dataset, limit=None):
    old_flags = None
    if name == "baseline_vector_only":
        old_flags = set_feature_flags(intent_filter=False, hyde=False)
        enable_rewrite = False
        enable_reflection = False
    elif name == "hybrid_bm25_vector":
        old_flags = set_feature_flags(intent_filter=False, hyde=False)
        enable_rewrite = False
        enable_reflection = False
    elif name == "plus_rerank":
        old_flags = set_feature_flags(intent_filter=False, hyde=False)
        enable_rewrite = False
        enable_reflection = False
    elif name == "plus_intent_filter":
        old_flags = set_feature_flags(intent_filter=True, hyde=False)
        enable_rewrite = False
        enable_reflection = False
    else:
        old_flags = set_feature_flags(intent_filter=True, hyde=True)
        enable_rewrite = True
        enable_reflection = True

    try:
        from src.agent.workflow import create_graph
        from src.llm_factory import create_chat_openai
        from src.retrieval.hybrid_search import HybridSearcher
        from src.retrieval.intent import classify_query_intent
        from src.retrieval.reranker import RerankProcessor
        from src.retrieval.vector_store import VectorStoreManager

        llm = create_chat_openai(temperature=0)
        vm = VectorStoreManager()
        retriever, _, _ = build_retriever(vm, name)
        app = create_graph(vm, retriever, llm, enable_rewrite=enable_rewrite, enable_reflection=enable_reflection)

        if limit:
            dataset = dataset.head(limit)
        rows = []
        for idx, row in dataset.iterrows():
            print(f"[{name}] {idx + 1}/{len(dataset)} {row['id']}")
            # 从 chat_history 中提取上一轮的 intent
            chat_history = parse_chat_history(row.get("chat_history"))
            prev_intent = None
            if chat_history:
                for item in reversed(chat_history):
                    if isinstance(item, dict) and item.get("role") == "user":
                        prev_intent = classify_query_intent(item.get("content", ""))
                        break
            started = time.perf_counter()
            final_state = await app.ainvoke({
                "query": row["question"],
                "chat_history": chat_history,
                "loop_step": 0,
                "prev_intent": prev_intent,
                "prev_documents": [],
            })
            rows.append(base_report_row(row, final_state, time.perf_counter() - started))
        report_df = pd.DataFrame(rows)
        summary = build_summary(report_df)
        summary.insert(0, "variant", name)
        return summary.iloc[0].to_dict()
    finally:
        if old_flags:
            restore_feature_flags(old_flags)


async def run_ablation(limit=None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(DATASET_PATH)
    variants = [
        "baseline_vector_only",
        "hybrid_bm25_vector",
        "plus_rerank",
        "plus_intent_filter",
        "full_pipeline",
    ]
    rows = []
    for variant in variants:
        row = await run_variant(variant, dataset, limit=limit)
        rows.append(row)
        print(row)
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    print("Saved ablation summary to", REPORT_DIR / "ablation_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only first N samples for smoke tests.")
    args = parser.parse_args()
    asyncio.run(run_ablation(limit=args.limit))
