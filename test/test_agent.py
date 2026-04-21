import sys
import os

# 1. 环境路径设置
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.reranker import RerankProcessor
from src.agent.workflow import create_graph 
from src.retrieval.hybrid_search import HybridSearcher
from langchain_openai import ChatOpenAI
from config import Config

def test_company_agent():
    print("--- 正在初始化系统 ---")
    vm=VectorStoreManager()
    hs=HybridSearcher(vm)
    base_retriever=hs.get_ensemble_retriever()
    reranker=RerankProcessor(base_retriever)
    
    llm=ChatOpenAI(
        model=Config.LLM_MODEL, 
        openai_api_key=Config.OPENAI_API_KEY, 
        openai_api_base=Config.OPENAI_BASE_URL,
        temperature=0
    )

    app = create_graph(vm, reranker,llm)
    inputs = {
    "query": "那鼠标呢？",
    "chat_history": [
        {"role": "user", "content": "办公用品申领范围包括哪些？"},
        {"role": "assistant", "content": "包括日常消耗品（笔、笔记本等）、耐用品（计算器、鼠标等）、打印耗材。"}
    ],
    "loop_step": 0
}
    print(f"\n用户问题: {inputs['query']}")
    print("-" * 30)

    for output in app.stream(inputs):
        for key,value in output.items():
            print(f"\n进入节点: [{key}]")
            if "rewrite_query" in value:
                print(f"   [改写结果]: {value['rewrite_query']}")
            if "documents" in value:
                print(f"   [检索到文档数量]: {len(value['documents'])}")
                for d in value['documents']:
                    print(f"   [实际内容片段]: {d.page_content[:50]}...") # 打印前50个字
            if "answer" in value:
                print(f"\n--- AI 最终回答 ---")
                print(value["answer"])
if __name__ == "__main__":
    test_company_agent()