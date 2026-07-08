from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate

from config import Config
from src.cache.redis_client import RedisCache

# 追问判定的简短标记词
_FOLLOW_UP_MARKERS = ("那", "这个", "如果", "超过", "多久", "也可以", "怎么办", "以后")


def _looks_like_follow_up(query: str) -> bool:
    stripped = (query or "").strip()
    return len(stripped) <= 18 or any(marker in stripped for marker in _FOLLOW_UP_MARKERS)


class QueryRewriter:
    # 查询重写→ 独立、完整的搜索词
    def __init__(self,llm):
        self.llm = llm
        self.cache = RedisCache() if Config.ENABLE_CACHE else None
        self.base_prompt = ChatPromptTemplate.from_template(
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
        # 追问专用改写 prompt：更强调补全上下文
        self.follow_up_prompt = ChatPromptTemplate.from_template(
            "你是企业知识库的检索优化专家。用户正在问一个追问式问题，请结合上一轮问答补全上下文。\n\n"
            "上一轮用户问题：{prev_question}\n"
            "上一轮回答摘要：{prev_answer}\n\n"
            "当前追问：{query}\n\n"
            "【改写要求】\n"
            "1. 请将追问补全为完整的独立查询句（例如「那超过30天呢？」→「病假超过30天的工资发放标准」）。\n"
            "2. 输出 2 条改写，每行 1 条，保留原问题中的数字条件和核心名词。\n"
            "3. 第 1 条是补全后的完整自然问句，第 2 条偏向制度/条款/流程等正式表述。\n"
            "4. 禁止引入原问题和新历史中没有的新事实。\n"
            "5. 不要关键词堆砌。\n"
            "\n改写："
        )

    @staticmethod
    def _extract_last_qa(chat_history: list) -> tuple[str, str]:
        """从对话历史中提取上一轮的 user question 和 assistant answer。"""
        prev_q = ""
        prev_a = ""
        if not chat_history:
            return prev_q, prev_a
        # 从后往前找最近的 user 和 assistant 对
        for item in reversed(chat_history):
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "assistant" and not prev_a:
                    prev_a = content
                elif role == "user" and not prev_q:
                    prev_q = content
            if prev_q and prev_a:
                break
        return prev_q, prev_a

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

        # 追问检测：如果是简短追问且有历史，使用上下文补全改写
        is_follow_up = _looks_like_follow_up(query) and bool(history_str.strip())

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

        if is_follow_up and isinstance(chat_history, list):
            prev_q, prev_a = self._extract_last_qa(chat_history)
            res = await self.llm.ainvoke(
                self.follow_up_prompt.format(
                    query=query,
                    prev_question=prev_q,
                    prev_answer=prev_a[:500] if prev_a else "",
                )
            )
        else:
            res = await self.llm.ainvoke(self.base_prompt.format(query=query, chat_history=history_str))

        variants = [query.strip()]
        for line in res.content.splitlines():
            candidate = line.strip().lstrip("-").strip()
            if not candidate:
                continue
            if candidate[0].isdigit() and ". " in candidate[:4]:
                candidate = candidate.split(". ", 1)[-1].strip()
            if candidate and candidate not in variants:#确保候选非空且不与已有查询重复（避免原始查询和改写完全一样）
                variants.append(candidate)

        max_variants = 3 if is_follow_up else 2
        variants = variants[:max_variants]
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

        # 追问检测：如果是简短追问且有历史，在 HyDE 中也注入上一轮 QA 上下文
        is_follow_up = _looks_like_follow_up(query) and bool(chat_history)

        if is_follow_up and isinstance(chat_history, list):
            prev_q, prev_a = self._extract_last_qa(chat_history)
            prompt = (
                "你是一位制度文档专家。请基于上一轮问答上下文，针对当前追问生成一段"
                "假设性的制度条款说明（用作 HyDE 假设文档检索）。\n\n"
                "上一轮用户问题：{prev_question}\n"
                "上一轮回答摘要：{prev_answer}\n\n"
                "当前追问（需补全理解）：{query}\n\n"
                "请生成一段流畅的制度条文，详细说明当前追问涉及的公司政策和规定。"
                "不要回答用户，直接输出制度正文。"
            ).format(prev_question=prev_q, prev_answer=prev_a[:500] if prev_a else "", query=query)
        else:
            prompt = (
                "你是一位制度文档专家。请针对以下问题，直接生成一段假设性的制度条款说明"
                "（用作 HyDE 假设文档检索），详细描述相关公司政策、流程和标准。"
                "不要回答用户，直接输出制度正文。\n\n"
                f"问题：{query}"
            )
        res = await self.llm.ainvoke(prompt)
        hyde = res.content.strip()
        if self.cache and hyde_cache_key:
            self.cache.set_json(hyde_cache_key, {"hyde": hyde}, expire=Config.HYDE_CACHE_TTL)
        return hyde
