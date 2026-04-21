from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import Config
class QueryRewriter:
    # 查询重写→ 独立、完整的搜索词
    def __init__(self,llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            "你是一个专业的搜索工程专家。请结合对话历史，将用户的问题改写为独立的搜索指令。\n"
            "【改写要求】：\n"
            "1. 识别问题涉及的核心业务领域（如行政办公、IT支持、人事制度）。\n"
            "2. 请结合对话历史，将用户的问题提取为 2-3 个最核心的搜索关键词，用空格隔开。不要输出指令性文字。例如：'办公用品 鼠标 申领'。\n"
            "3. 不要输出多余的指令性文字，关键词间用空格隔开。\n"
            "\n历史：{chat_history}\n"
            "问题：{query}"
        )

    def rewrite(self, query: str, chat_history: Optional[Union[str, List[dict]]] = None) -> str:
        if isinstance(chat_history, list):
            history_str = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                for msg in chat_history
            )
        elif isinstance(chat_history, str):
            history_str = chat_history
        else:
            history_str = ""

        # 调用 LLM，传入正确的 history_str
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query, "chat_history": history_str})
        return response.content.strip()