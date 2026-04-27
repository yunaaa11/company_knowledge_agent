import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from config import Config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.reranker import RerankProcessor
from src.retrieval.hybrid_search import HybridSearcher
from src.agent.workflow import create_graph
from src.evaluation.test_data import TestDataGenerator
from src.evaluation.ragas_metrics import RagasEvaluator
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, UnstructuredWordDocumentLoader
async def  run_evaluation_pipeline():
    # --- 1. 初始化组件 ---
    llm=ChatOpenAI(model=Config.LLM_MODEL, temperature=0)
    embeddings =HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    vm=VectorStoreManager()
    hs=HybridSearcher(vm)
    base_retriever=hs.get_ensemble_retriever()
    reranker=RerankProcessor(base_retriever)
    app=create_graph(vm,reranker,llm)


 # --- 2. 手动构造测试集（基于6 个文档内容）---
    print("--- 准备测试用例（基于公司制度文档）---")
    testset_df = pd.DataFrame({
        "question": [
            # 员工请假管理制度
            "员工请病假需要提供什么材料？病假期间的工资如何发放？",
            "年假的享受天数与工龄有什么关系？",
            # 报销管理制度
            "在一线城市出差，住宿标准是多少？报销的审批流程需要多长时间？",
            # IT 故障处理指南
            "无法连接公司 WiFi 应该怎么办？",
            # 办公用品申领流程
            "新员工入职可以申领哪些办公用品？",
            # 信息安全行为规范
            "公司对员工密码有哪些具体要求？",
            # 绩效管理制度
            "绩效考核结果分为哪几个等级？连续两个季度 D 级会有什么后果？"
        ],
        "ground_truth": [
            "病假需提供医院开具的病假证明，病假期间发放基本工资的 80%。",
            "入职满 1 年享 5 天带薪年假，满 5 年享 10 天，满 10 年享 15 天。",
            "一线城市（北京/上海/广州/深圳）住宿每晚不超过 600 元。报销流程：上级审批 2 个工作日，财务审核 3 个工作日，打款 5 个工作日。",
            "1.确认密码正确；2.忘记网络后重连；3.重启电脑和路由器；4.联系 IT 分机 8888。",
            "可领取笔记本 2 本、笔 3 支、文件夹 1 个、鼠标垫 1 个。",
            "密码长度至少 8 位，包含大小写字母、数字、特殊符号，每 90 天更换一次，禁止弱密码。",
            "考核结果分为 S、A、B、C、D 五级。连续两个季度 D 级进入观察期，连续三个季度 D 级解除劳动合同。"
        ]
    })
    print(f"使用基于文档内容的手动测试集，共 {len(testset_df)} 个问题")



    # --- 3. 运行 Agent 并收集结果 ---
    print("--- 正在运行 Agent 获取回答 ---")
    evaluator=RagasEvaluator(llm)
    all_results=[]
    
    for _,row in testset_df.iterrows():
        question=row['question']
        ground_truth=row['ground_truth']
        # 模拟 Agent 流程
        inputs = {"query": question, "chat_history": [], "loop_step": 0}
        final_state = await app.ainvoke(inputs)          # 直接获取最终状态（包含所有字段）
        # 提取 answer 和 documents
        node_output = {
    "answer": final_state.get("answer", ""),
    "documents": final_state.get("documents", [])
}
        # 执行 RAGAS 评分
        score_df=evaluator.evaluate_response(question,node_output,ground_truth)
        all_results.append(score_df)
    # --- 4. 输出汇总报告 ---
    final_report=pd.concat(all_results)
    print("\n评估结果汇总:")
    print(final_report[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])
    final_report.to_csv("eval_report.csv", index=False)
    print("\n评估报告已保存至 eval_report.csv")
if __name__ == "__main__":
    asyncio.run(run_evaluation_pipeline())