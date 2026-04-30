import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from config import Config
from langchain_openai import ChatOpenAI
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.reranker import RerankProcessor
from src.retrieval.hybrid_search import HybridSearcher
from src.agent.workflow import create_graph
from src.evaluation.ragas_metrics import RagasEvaluator

def build_eval_dataset() -> pd.DataFrame:
    """
    评估集尽量贴近真实业务问题，覆盖：
    1. 单文档事实问答
    2. 多条件/多子问题
    3. 流程步骤题
    4. 跨制度组合题
    5. 口语化表达
    6. 多轮追问改写能力
    7. 无答案/拒答能力
    """
    return pd.DataFrame([
        {
            "category": "single_doc_fact",
            "question": "员工请病假需要提供什么材料？病假期间工资怎么发？",
            "ground_truth": "病假需提供医院开具的病假证明，病假期间发放基本工资的 80%。",
        },
        {
            "category": "single_doc_fact",
            "question": "年假天数和工龄怎么对应？",
            "ground_truth": "入职满 1 年享 5 天带薪年假，满 5 年享 10 天，满 10 年享 15 天。",
        },
        {
            "category": "process_step",
            "question": "一线城市出差住宿标准是多少？报销审批和打款分别要多久？",
            "ground_truth": "一线城市（北京/上海/广州/深圳）住宿每晚不超过 600 元。报销流程：上级审批 2 个工作日，财务审核 3 个工作日，打款 5 个工作日。",
        },
        {
            "category": "process_step",
            "question": "电脑连不上公司 WiFi，一般先怎么排查？最后找谁？",
            "ground_truth": "1.确认密码正确；2.忘记网络后重连；3.重启电脑和路由器；4.联系 IT 分机 8888。",
        },
        {
            "category": "single_doc_fact",
            "question": "新员工刚入职能领哪些办公用品？数量是多少？",
            "ground_truth": "可领取笔记本 2 本、笔 3 支、文件夹 1 个、鼠标垫 1 个。",
        },
        {
            "category": "constraint_rule",
            "question": "公司密码至少要满足哪些要求？多久换一次？",
            "ground_truth": "密码长度至少 8 位，包含大小写字母、数字、特殊符号，每 90 天更换一次，禁止弱密码。",
        },
        {
            "category": "single_doc_fact",
            "question": "绩效结果有哪些等级？连续两个季度 D 会怎么样？",
            "ground_truth": "考核结果分为 S、A、B、C、D 五级。连续两个季度 D 级进入观察期，连续三个季度 D 级解除劳动合同。",
        },
        {
            "category": "cross_doc",
            "question": "员工因为生病请假时，除了请假材料外，还要不要遵守密码更新要求？",
            "ground_truth": "病假需提供医院开具的病假证明，病假期间发放基本工资的 80%。同时公司密码要求至少 8 位，包含大小写字母、数字、特殊符号，并且每 90 天更换一次。",
        },
        {
            "category": "multi_constraint",
            "question": "连续两个季度绩效 D 级的员工，如果又要去一线城市出差，住宿标准和绩效后果分别是什么？",
            "ground_truth": "连续两个季度 D 级进入观察期。一线城市（北京/上海/广州/深圳）住宿每晚不超过 600 元。",
        },
        {
            "category": "colloquial_query",
            "question": "我密码老提示太弱，到底得怎么设才行？",
            "ground_truth": "密码长度至少 8 位，包含大小写字母、数字、特殊符号，每 90 天更换一次，禁止弱密码。",
        },
        {
            "category": "follow_up",
            "question": "那审批完以后多久能打款？",
            "ground_truth": "报销流程中，上级审批 2 个工作日，财务审核 3 个工作日，打款 5 个工作日。",
            "chat_history": [
                {"role": "user", "content": "在一线城市出差住宿标准是多少？"},
                {"role": "assistant", "content": "一线城市住宿每晚不超过 600 元。"},
            ],
        },
        {
            "category": "no_answer",
            "question": "公司有没有规定结婚礼金报销标准？",
            "ground_truth": "资料中未找到相关信息。",
        },
    ])

async def  run_evaluation_pipeline():
    # --- 1. 初始化组件 ---
    llm=ChatOpenAI(model=Config.LLM_MODEL, temperature=0)

    vm=VectorStoreManager()
    hs=HybridSearcher(vm)
    base_retriever=hs.get_ensemble_retriever()
    reranker=RerankProcessor(base_retriever)
    app=create_graph(vm,reranker,llm)


    # --- 2. 手动构造测试集（贴近真实业务）---
    print("--- 准备测试用例（覆盖多类真实业务问题）---")
    testset_df = build_eval_dataset()
    print(f"使用手动评估集，共 {len(testset_df)} 个问题")
    print("类别分布:")
    print(testset_df["category"].value_counts())



    # --- 3. 运行 Agent 并收集结果 ---
    print("--- 正在运行 Agent 获取回答 ---")
    evaluator = RagasEvaluator(llm, embeddings=vm.embeddings)
    all_results=[]
    
    for idx, row in testset_df.iterrows():
        question=row["question"]
        ground_truth=row["ground_truth"]
        chat_history = row.get("chat_history", [])
        print(f"\n[{idx + 1}/{len(testset_df)}] [{row['category']}] 正在评估问题: {question}")
        #  批量评估循环 模拟 Agent 流程
        inputs = {"query": question, "chat_history": chat_history, "loop_step": 0}
        final_state = await app.ainvoke(inputs)          # 直接获取最终状态（包含所有字段）
        # 提取 answer 和 documents
        node_output = {
            "answer": final_state.get("answer", ""),
            "documents": final_state.get("documents", [])
        }
        # 执行 RAGAS 评分
        score_df=evaluator.evaluate_response(question,node_output,ground_truth)
        score_df["category"] = row["category"]
        score_df["rewrite_query"] = " | ".join(final_state.get("rewrite_query", [])) if isinstance(final_state.get("rewrite_query"), list) else final_state.get("rewrite_query", "")
        score_df["loop_step"] = final_state.get("loop_step", 0)
        all_results.append(score_df)
    # --- 4. 输出汇总报告 ---
    final_report=pd.concat(all_results)
    summary_df = pd.DataFrame([{
        "sample_count": len(final_report),
        "avg_faithfulness": round(final_report["faithfulness"].mean(), 4),
        "avg_answer_relevancy": round(pd.to_numeric(final_report["answer_relevancy"], errors="coerce").fillna(0).mean(), 4),
        "avg_context_precision": round(final_report["context_precision"].mean(), 4),
        "avg_context_recall": round(final_report["context_recall"].mean(), 4),
        "avg_strict_score": round(final_report["strict_score"].mean(), 4),
        "avg_retrieval_count": round(final_report["retrieval_count"].mean(), 2),
    }])

    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "eval_report.csv")
    summary_path = os.path.join(report_dir, "eval_summary.csv")

    print("\n评估结果汇总:")
    print(final_report[[
        "question",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "top_relevance_score",
        "strict_score",
    ]])
    print("\n评估指标均值:")
    print(summary_df)
    print("\n按类别统计的严格分:")
    print(
        final_report.groupby("category", dropna=False)["strict_score"]
        .mean()
        .round(4)
        .sort_values(ascending=False)
    )

    final_report.to_csv(report_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n评估报告已保存至 {report_path}")
    print(f"评估摘要已保存至 {summary_path}")
if __name__ == "__main__":
    asyncio.run(run_evaluation_pipeline())