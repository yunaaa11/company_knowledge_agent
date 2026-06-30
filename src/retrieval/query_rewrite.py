from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate

from config import Config
from src.cache.redis_client import RedisCache

class QueryRewriter:
    # 查询重写→ 独立、完整的搜索词
    def __init__(self,llm):
        self.llm = llm
        self.cache = RedisCache() if Config.ENABLE_CACHE else None
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
        rewrite_cache_key = None
        if self.cache:
            rewrite_cache_key = self.cache.generate_stage_key(
                stage="rewrite",
                query=query,
                chat_history=chat_history,
                index_version=Config.INDEX_VERSION,
                prompt_version=Config.PROMPT_VERSION,
                prefix=Config.CACHE_KEY_PREFIX,
            )
            cached = self.cache.get_json(rewrite_cache_key)
            if cached and cached.get("rewrite_query"):
                return cached["rewrite_query"]

        res = await self.llm.ainvoke(self.prompt.format(query=query, chat_history=history_str))

        variants = [query.strip()]
        for line in res.content.splitlines():
            candidate = line.strip().lstrip("-").strip()
            if not candidate:
                continue
            if candidate[0].isdigit() and ". " in candidate[:4]:
                candidate = candidate.split(". ", 1)[-1].strip()
            if candidate and candidate not in variants:#确保候选非空且不与已有查询重复（避免原始查询和改写完全一样）
                variants.append(candidate)

        variants = variants[:3]
        if self.cache and rewrite_cache_key:
            self.cache.set_json(rewrite_cache_key, {"rewrite_query": variants}, expire=Config.REWRITE_CACHE_TTL)
        return variants

    async def generate_hyde(self, query: str, chat_history=None) -> str:
        """假设文档生成 扮演专家，直接生成一段针对该问题的“假设性完美答案”。这段生成的答案不直接回复用户，而是作为检索词去向量库中找语义相似的原文"""
        if not Config.ENABLE_HYDE:
            return ""
        hyde_cache_key = None
        if self.cache:
            hyde_cache_key = self.cache.generate_stage_key(
                stage="hyde",
                query=query,
                chat_history=chat_history,
                index_version=Config.INDEX_VERSION,
                prompt_version=Config.PROMPT_VERSION,
                prefix=Config.CACHE_KEY_PREFIX,
            )
            cached = self.cache.get_json(hyde_cache_key)
            if cached and cached.get("hyde"):
                return cached["hyde"]

        prompt = (
            "???????????????????????????????????????"
            "???????????????????????????????????\n"
            f"?????{query}"
        )
        res = await self.llm.ainvoke(prompt)
        hyde = res.content.strip()
        if self.cache and hyde_cache_key:
            self.cache.set_json(hyde_cache_key, {"hyde": hyde}, expire=Config.HYDE_CACHE_TTL)
        return hyde
