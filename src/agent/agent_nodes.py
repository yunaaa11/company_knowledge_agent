import asyncio

from langchain_core.documents import Document

from config import Config
from src.cache.redis_client import RedisCache
from src.retrieval.query_rewrite import QueryRewriter
from src.retrieval.reranker import RerankProcessor
from src.retrieval.intent import classify_query_intent
from src.agent.states import AgentState
import os

class Nodes:
    def __init__(self,vector_manager,reranker,llm):
        self.rewriter=QueryRewriter(llm=llm)
        self.reranker=reranker
        self.llm = llm
        self.max_fused_docs = Config.MAX_FUSED_DOCS
        self.cache = RedisCache() if Config.ENABLE_CACHE else None


    @staticmethod
    def _doc_to_cache(doc):
        return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}

    @staticmethod
    def _doc_from_cache(item):
        return Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}) or {})

    @staticmethod
    def _doc_source(doc, rank):
        metadata = doc.metadata or {}
        score = metadata.get("relevance_score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        return {
            "rank": rank,
            "source": metadata.get("source", "unknown"),
            "relevance_score": score,
            "snippet": " ".join((doc.page_content or "").split())[:300],
            "page_content": doc.page_content,
        }

    def _retrieve_once(self, query, prev_intent=None):
        if hasattr(self.reranker, "retrieve"):
            return self.reranker.retrieve(query, prev_intent=prev_intent)
        return self.reranker.invoke(query)

    async def rewrite_node(self,state:AgentState):
        print("--- 正在改写问题 ---")
        chat_history = state.get("chat_history")   # 从状态中获取历史
        rewrite_task = self.rewriter.rewrite(state["query"], chat_history=chat_history)
        hyde_task = self.rewriter.generate_hyde(state["query"], chat_history=chat_history)
        new_query, hyde_query = await asyncio.gather(rewrite_task, hyde_task)
        # 检测当前问题的意图并保存为 prev_intent
        detected_intent = classify_query_intent(
            state["query"],
            chat_history=chat_history,
            prev_intent=state.get("prev_intent"),
        )
        print(f"--- 当前意图: {detected_intent} | 上一轮意图: {state.get('prev_intent')} ---")
        return {
            "rewrite_query": new_query,
            "hyde_query": hyde_query,
            "loop_step": state.get("loop_step", 0) + 1,
            "prev_intent": detected_intent or state.get("prev_intent"),
        }

    async def retrieve_node(self,state:AgentState):
        """多查询并行检索 + 缓存 + 去重融合  加权多查询融合"""
        print("--- retrieving documents ---")
        rewrite_queries = state.get("rewrite_query") or state.get("query", "")
        if isinstance(rewrite_queries, str):
            rewrite_queries = [rewrite_queries]

        queries = []
        seen_query = set()
        #区分三种查询来源：- original（原始问题）- rewrite（改写后问题）- hyde（HyDE 生成的假设文档）
        def add_query(query, weight, kind):
            if query and query not in seen_query:
                queries.append({"query": query, "weight": weight, "kind": kind})
                seen_query.add(query)

        add_query(state.get("query", ""), Config.ORIGINAL_QUERY_WEIGHT, "original")
        for query in rewrite_queries:
            add_query(query, Config.REWRITTEN_QUERY_WEIGHT, "rewrite")
        if Config.ENABLE_HYDE:
            add_query(state.get("hyde_query"), Config.HYDE_QUERY_WEIGHT, "hyde")

        # 跨文档联合检索：如果当前意图涉及跨文档场景，自动并行检索多个 policy type
        prev_intent = state.get("prev_intent")
        cache_key = None
        if self.cache:
            cache_key = self.cache.generate_stage_key(
                stage="retrieval",
                query=" | ".join(item["query"] for item in queries),
                chat_history=state.get("chat_history"),
                index_version=Config.INDEX_VERSION,
                prompt_version=Config.PROMPT_VERSION,
                prefix=Config.CACHE_KEY_PREFIX,
            )
            cached = self.cache.get_json(cache_key)
            #缓存命中分支
            if cached and cached.get("documents"):
                docs = [self._doc_from_cache(item) for item in cached["documents"]]
                sources = [self._doc_source(doc, idx) for idx, doc in enumerate(docs, start=1)]
                cached_stats = cached.get("retrieval_query_stats", {})
                return {
                    "documents": docs,
                    "retrieval_sources": sources,
                    "retrieval_cache_hit": True,
                    "retrieval_query_stats": cached_stats,
                }

        tasks = [asyncio.to_thread(self._retrieve_once, item["query"], prev_intent) for item in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored_docs = {}
        subquery_stats = []
        for item, result in zip(queries, results):
            query = item["query"]
            weight = item["weight"]
            kind = item["kind"]
            if isinstance(result, Exception):
                print(f"retrieval failed for query='{query}': {result}")
                docs = []
            else:
                docs = result or []
            subquery_stats.append((kind, query, len(docs)))
            #按 fingerprint 去重，保留加权分数最高的文档（weighted_score 最大者）
            #如果同一文档被多个查询召回，但得分不同（例如改写查询的权重更高），新版会保留加权后得分最高的版本，更符合多查询融合的常见策略。
            for doc in docs:
                fingerprint = (doc.metadata.get("source", ""), doc.page_content[:200])
                base_score = doc.metadata.get("relevance_score", 0.5) or 0.5
                try:
                    base_score = float(base_score)
                except (TypeError, ValueError):
                    base_score = 0.5
                weighted_score = base_score * weight
                if fingerprint in scored_docs and scored_docs[fingerprint].metadata.get("weighted_score", 0) >= weighted_score:
                    continue
                doc.metadata["weighted_score"] = weighted_score
                #kind 主要用于 日志追踪 和 元数据标记
                doc.metadata["query_kind"] = kind
                scored_docs[fingerprint] = doc
        #缓存未命中分支
        all_docs = list(scored_docs.values())
        all_docs.sort(
            key=lambda doc: doc.metadata.get("weighted_score", doc.metadata.get("relevance_score", 0.0)),
            reverse=True,
        )
        #截断、构建来源、统计、缓存存储和日志打印
        all_docs = all_docs[:self.max_fused_docs]
        sources = [self._doc_source(doc, idx) for idx, doc in enumerate(all_docs, start=1)]
        retrieval_query_stats = {
            "raw": [
                {"kind": kind, "query": query, "count": count}
                for kind, query, count in subquery_stats
            ],
            "counts_by_kind": {},
        }
        for kind, _, count in subquery_stats:
            retrieval_query_stats["counts_by_kind"][kind] = retrieval_query_stats["counts_by_kind"].get(kind, 0) + count

        if self.cache and cache_key and all_docs:
            self.cache.set_json(
                cache_key,
                {
                    "documents": [self._doc_to_cache(doc) for doc in all_docs],
                    "retrieval_query_stats": retrieval_query_stats,
                },
                expire=Config.RETRIEVAL_CACHE_TTL,
            )

        if subquery_stats:
            print("subquery retrieval summary:")
            for idx, (kind, _, count) in enumerate(subquery_stats, start=1):
                print(f"  - query {idx} ({kind}): {count} docs")
        print(f"final fused docs: {len(all_docs)}")
        return {
            "documents": all_docs,
            "retrieval_sources": sources,
            "retrieval_cache_hit": False,
            "retrieval_query_stats": retrieval_query_stats,
        }

    async def generate_node(self,state:AgentState):
        print("--- 正在生成回答 ---")
        context="\n".join([d.page_content for d in state["documents"]])
        rewritten = state.get("rewrite_query") or state.get("query", "")
        if isinstance(rewritten, list):
            rewritten = " | ".join(rewritten)
        system_prompt = (
    "你是一个严谨的企业行政助手，负责回答员工关于公司制度的问题。\n"
    "请严格依据以下资料回答问题，不得编造或使用外部知识。\n\n"
    "【可用制度文档】\n"
    "1. 《员工请假管理制度》（人力资源部）\n"
    "2. 《员工报销管理制度》（财务部）\n"
    "3. 《IT 故障处理指南》（IT部）\n"
    "4. 《办公用品申领流程》（行政部）\n"
    "5. 《信息安全行为规范》（IT部/合规部）\n"
    "6. 《员工绩效管理制度》（人力资源部）\n\n"
    "【冲突处理规则】\n"
    "如果不同部门/文档对同一事项的规定存在冲突，请按以下优先级采纳：\n"
    "- 最高优先级：公司层面的强制性规范（如信息安全行为规范）\n"
    "- 次优先级：人力资源部发布的制度（请假、绩效）\n"
    "- 第三优先级：财务部（报销）、行政部（办公用品）\n"
    "- 最低优先级：IT部操作指南（仅作参考，不与其他部门强制性规则冲突）\n"
    "若无法判断优先级，请如实列出不同规定，并提示用户以最新发布的正式制度为准。\n\n"
    "【回答要求 — 严格遵循】\n"
    "1. 引用具体条款时，注明来源文档名称（例如：根据《员工请假管理制度》第四条）。\n"
    "2. 禁止添加任何常识、推测或外部知识。你的回答必须严格基于上方【资料】原文。\n"
    "3. 如果检索到的资料不足以回答问题，请明确说\"资料中未找到相关信息\"，不要尝试补充。\n"
    "4. 回答应简洁、结构化，可使用分点或表格帮助理解。\n"
    "5. 对于涉及金额、天数、百分比等具体数字，务必核对准确。\n"
    "6. 禁止给出超出制度范围的建议（如\"可以申请更多年假\"）。\n"
    "7. 如果问题包含多个子问题，必须逐项覆盖，不能漏答。\n"
    "8. 若上下文中存在相互矛盾的信息，先说明冲突，再按优先级给出结论。\n"
    "9. 只允许复述文档中明确出现的规则，禁止补充文档未出现的条件、阈值或例外。\n"
    "10. 禁止使用\"通常\"\"一般\"\"可能\"\"建议\"等推测性表述。\n\n"
    "【正例】\n"
    "- 资料原文：\"病假超过30天需提供三甲医院证明\"\n"
    "  ✅ 正确回答：\"根据规定，病假超过30天需要提供三甲医院证明。\"\n"
    "- 资料原文未提及工资发放\n"
    "  ✅ 正确回答：\"资料中未找到病假工资发放的相关信息。\"\n\n"
    "【反例】\n"
    "- 资料未提及金额\n"
    "  ❌ 错误回答：\"出差补贴一般为每天200元。\"（注：补充了外部知识）\n"
    "- 资料只说了病假天数\n"
    "  ❌ 错误回答：\"建议您尽快提交病假申请。\"（注：使用了建议性表述）\n"
)
        prompt = f"{system_prompt}\n\n根据资料：{context} 回答：{rewritten}"
        response = await self.llm.ainvoke(prompt)
        return {"answer": response.content}
