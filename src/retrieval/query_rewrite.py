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
            "2. 请将用户的问题改写为一个完整的、独立的中文问句，不要输出关键词列表,例如：'鼠标属于哪类办公用品？耐用品鼠标的申领标准和流程是什么？'。\n"
            "3. 不要输出多余的指令性文字，关键词间用空格隔开。\n"
            "\n历史：{chat_history}\n"
            "问题：{query}"
        )

    async def rewrite(self, query: str, chat_history: Optional[Union[str, List[dict]]] = None) -> str:
        # 处理历史记录格式
        if isinstance(chat_history, list):
            history_str = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                for msg in chat_history
            )
        elif isinstance(chat_history, str):
            history_str = chat_history
        else:
            history_str = ""

        # 调用 LLM
        res=await self.llm.ainvoke(self.prompt.format(query=query,chat_history=history_str))
        # 按行切分并清理
        raw_queries=[q.strip() for q in res.content.split("\n") if q.strip()]
        # 排除掉可能出现的“1. ”、“2. ”前缀
        clean_queries=[q.split(".",1)[-1] if q[0].isdigit() else q for q in raw_queries]
        # 确保包含原始查询，并取前 3 个
        return clean_queries[:3]