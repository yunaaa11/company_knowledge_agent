import sys
import os
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# 设置路径
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from src.agent.reflection import Reflection, Grade
from config import Config

async def test_reflection_logic():
    print("--- 开始测试 Reflection 2.0 逻辑 ---")
    
    # 1. 初始化 LLM
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        openai_api_base=Config.OPENAI_BASE_URL,
        temperature=0
    )

    # ---------------------------------------------------------
    # 场景 A：高质量检索（预期结果：generate）
    # ---------------------------------------------------------
    state_good = {
        "query": "电脑坏了怎么报修？",
        "rewrite_query": "IT设备故障报修流程",
        "documents": [
            Document(page_content="新员工入职可领取笔记本电脑。如遇硬件故障，请联系IT部，拨打分机8888报修。"),
            Document(page_content="IT部负责所有办公设备的维修维护工作。")
        ],
        "loop_step": 1
    }
    
    print("\n测试场景 A：文档与问题高度相关...")
    res_a = await Reflection.grade_documents_complex(state_good, llm)
    print(f"结果: {res_a} (预期: generate)")

    # ---------------------------------------------------------
    # 场景 B：低质量检索/噪声（预期结果：retry）
    # ---------------------------------------------------------
    state_bad = {
        "query": "怎么申请笔记本？",
        "rewrite_query": "办公用品申领标准",
        "documents": [
            Document(page_content="食堂开餐时间为中午11:30到1:30。"),
            Document(page_content="严禁在办公区吸烟，违者罚款。")
        ],
        "loop_step": 1
    }
    
    print("\n测试场景 B：文档全是无关噪声...")
    res_b = await Reflection.grade_documents_complex(state_bad, llm)
    print(f"结果: {res_b} (预期: retry)")

    # ---------------------------------------------------------
    # 场景 C：达到最大重试次数（预期结果：generate - 强行终止循环）
    # ---------------------------------------------------------
    state_limit = state_bad.copy()
    state_limit["loop_step"] = 3 # 假设最大步数是3
    
    print("\n测试场景 C：质量虽差但已达到循环上限...")
    res_c = await Reflection.grade_documents_complex(state_limit, llm)
    print(f"结果: {res_c} (预期: generate)")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_reflection_logic())