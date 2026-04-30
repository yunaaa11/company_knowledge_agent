import asyncio
import os
import sys
from typing import Dict, List

import pandas as pd
from langchain_openai import ChatOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.agent.workflow import create_graph
from src.evaluation.ragas_metrics import RagasEvaluator
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import RerankProcessor
from src.retrieval.vector_store import VectorStoreManager
from run_eval import build_eval_dataset


def _build_app(vm: VectorStoreManager, llm, use_rerank: bool, enable_rewrite: bool, enable_reflection: bool):
    hs = HybridSearcher(vm)
    base_retriever = hs.get_ensemble_retriever()
    #根据 use_rerank 决定是否在检索后附加重排序器
    retriever = RerankProcessor(base_retriever) if use_rerank else base_retriever
    return create_graph(
        vm,
        retriever,
        llm,
        enable_rewrite=enable_rewrite,
        enable_reflection=enable_reflection,
    )


async def _run_one_variant(
    name: str,
    vm: VectorStoreManager,
    llm,
    testset_df: pd.DataFrame,
    use_rerank: bool,
    enable_rewrite: bool,
    enable_reflection: bool,
) -> Dict[str, float]:
    app = _build_app(vm, llm, use_rerank, enable_rewrite, enable_reflection)
    evaluator = RagasEvaluator(llm, embeddings=vm.embeddings)
    rows: List[pd.DataFrame] = []

    for _, row in testset_df.iterrows():
        inputs = {
            "query": row["question"],
            "chat_history": row.get("chat_history", []),
            "loop_step": 0,
        }
        #调用 app.ainvoke 获得最终状态（包含 answer 和 documents）。
        final_state = await app.ainvoke(inputs)
        node_output = {
            "answer": final_state.get("answer", ""),
            "documents": final_state.get("documents", []),
        }
        #使用 RagasEvaluator 计算指标
        score_df = evaluator.evaluate_response(row["question"], node_output, row["ground_truth"])
        rows.append(score_df)

    # 收集所有样本的评分 DataFrame，合并后计算各项指标的均值。
    # 返回一个字典，包含变体名称及各指标平均值。
    final_report = pd.concat(rows, ignore_index=True)
    return {
        "variant": name,
        "sample_count": len(final_report),
        "avg_faithfulness": round(pd.to_numeric(final_report["faithfulness"], errors="coerce").fillna(0).mean(), 4),
        "avg_answer_relevancy": round(pd.to_numeric(final_report["answer_relevancy"], errors="coerce").fillna(0).mean(), 4),
        "avg_context_precision": round(pd.to_numeric(final_report["context_precision"], errors="coerce").fillna(0).mean(), 4),
        "avg_context_recall": round(pd.to_numeric(final_report["context_recall"], errors="coerce").fillna(0).mean(), 4),
        "avg_strict_score": round(pd.to_numeric(final_report["strict_score"], errors="coerce").fillna(0).mean(), 4),
        "avg_retrieval_count": round(pd.to_numeric(final_report["retrieval_count"], errors="coerce").fillna(0).mean(), 2),
    }


async def run_ablation():
    llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0)
    vm = VectorStoreManager()
    #测试集来源
    testset_df = build_eval_dataset()
     
    #四种变体：baseline 无额外优化，仅基础检索+生成   plus_rewrite 仅增加查询改写 
    #plus_rerank 增加改写+重排序 plus_reflection(full) 全功能（改写+重排+反思重试
    variants = [
        ("baseline(no_rewrite_no_rerank_no_reflection)", False, False, False),
        ("plus_rewrite", False, True, False),
        ("plus_rerank", True, True, False),
        ("plus_reflection(full)", True, True, True),
    ]
    summary_rows = []
    for name, use_rerank, enable_rewrite, enable_reflection in variants:
        print(f"\n=== Running variant: {name} ===")
        row = await _run_one_variant(
            name=name,
            vm=vm,
            llm=llm,
            testset_df=testset_df,
            use_rerank=use_rerank,
            enable_rewrite=enable_rewrite,
            enable_reflection=enable_reflection,
        )
        print(row)
        summary_rows.append(row)

    ablation_df = pd.DataFrame(summary_rows)
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "ablation_summary.csv")
    ablation_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nAblation summary saved to: {out_path}")
    print(ablation_df)


if __name__ == "__main__":
    asyncio.run(run_ablation())

