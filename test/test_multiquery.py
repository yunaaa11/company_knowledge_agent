import asyncio
import sys
import os

# 设置路径
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from src.retrieval.query_rewrite import QueryRewriter
from langchain_openai import ChatOpenAI
from config import Config
async def test_mq():
    print("--- 开始测试多策略改写 ---")
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        openai_api_base=Config.OPENAI_BASE_URL,
        temperature=0
    )
    rewriter=QueryRewriter(llm)
    query="笔记本电脑坏了怎么修？"
    history=[{"role":"user","content":"我想领一个办公用品"}]
    # 1. 测试改写
    queries = await rewriter.rewrite(query, chat_history=history)
    print(f"\n原始问题: {query}")
    print("生成的 3 个搜索策略:")
    for i,q in enumerate(queries):
        print(f" {i+1}.{q}")
    #验证是否返回了列表
    assert isinstance(queries,list)#如果返回的不是列表，或者列表为空，assert 会抛出 AssertionError
    assert len(queries)>0
if __name__ == "__main__":
    asyncio.run(test_mq())