from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate

class QueryRewriter:
    # 查询重写→ 独立、完整的搜索词
    def __init__(self,llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            "你是企业知识库的检索优化专家。请结合对话历史，为用户问题生成适合制度文档检索的查询变体。\n"
            "【改写要求】\n"
            "1. 输出 2 条改写，每行 1 条，不要编号，不要解释。\n"
            "2. 每条都必须保留原问题中的核心约束、关键名词、数字条件和部门/制度名称。\n"
            "3. 第 1 条偏向用户原意的完整问句，第 2 条偏向制度/条款/流程/标准等正式表述。\n"
            "4. 如果原问题已经足够清晰，就输出与原问题高度近似的检索问句，不要过度改写。\n"
            "5. 禁止输出关键词堆砌，禁止引入原问题中没有的新事实。\n"
            "\n历史：{chat_history}\n"
            "问题：{query}"
        )

    async def rewrite(self, query: str, chat_history: Optional[Union[str, List[dict]]] = None) -> List[str]:
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
        res = await self.llm.ainvoke(self.prompt.format(query=query, chat_history=history_str))

        variants = [query.strip()]
        for line in res.content.splitlines():
            candidate = line.strip().lstrip("-").strip()
            if not candidate:
                continue
            if candidate[0].isdigit() and ". " in candidate[:4]:
                candidate = candidate.split(". ", 1)[-1].strip()
            if candidate and candidate not in variants:
                variants.append(candidate)

        return variants[:3]