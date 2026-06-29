import asyncio

from langchain_core.documents import Document

from config import Config
from src.cache.redis_client import RedisCache
from src.retrieval.query_rewrite import QueryRewriter
from src.retrieval.reranker import RerankProcessor
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

    def _retrieve_once(self, query):
        if hasattr(self.reranker, "retrieve"):
            return self.reranker.retrieve(query)
        return self.reranker.invoke(query)

    async def rewrite_node(self,state:AgentState):
        print("--- 正在改写问题 ---")
        chat_history = state.get("chat_history")   # 从状态中获取历史
        new_query=await self.rewriter.rewrite(state["query"], chat_history=chat_history)
        return {"rewrite_query": new_query, "loop_step": state.get("loop_step", 0) + 1}
    #多查询并行检索 + 缓存 + 去重融合
    async def retrieve_node(self,state:AgentState):
        print("--- retrieving documents ---")
        #获取查询列表
        queries = state.get("rewrite_query") or state.get("query", "")
        if isinstance(queries, str):
            queries = [queries]
        #缓存检查
        cache_key = None
        if self.cache:
            cache_key = self.cache.generate_stage_key(
                stage="retrieval",
                query=" | ".join(queries),
                chat_history=state.get("chat_history"),
                index_version=Config.INDEX_VERSION,
                prompt_version=Config.PROMPT_VERSION,
                prefix=Config.CACHE_KEY_PREFIX,
            )
            cached = self.cache.get_json(cache_key)
            if cached and cached.get("documents"):
                docs = [self._doc_from_cache(item) for item in cached["documents"]]
                sources = [self._doc_source(doc, idx) for idx, doc in enumerate(docs, start=1)]
                return {"documents": docs, "retrieval_sources": sources, "retrieval_cache_hit": True}
        #并行检索
        tasks = [asyncio.to_thread(self._retrieve_once, q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
         #结果合并与去重
        all_docs = []
        seen_sources = set()
        subquery_stats = []
        for q, result in zip(queries, results):
            if isinstance(result, Exception):
                print(f"retrieval failed for query='{q}': {result}")
                docs = []
            else:
                docs = result or []
            subquery_stats.append((q, len(docs)))
            for doc in docs:
                fingerprint = (doc.metadata.get("source", ""), doc.page_content[:200])
                if fingerprint in seen_sources:
                    continue
                seen_sources.add(fingerprint)
                all_docs.append(doc)
        #排序与截断
        all_docs.sort(key=lambda doc: doc.metadata.get("relevance_score", 0.0), reverse=True)
        all_docs = all_docs[:self.max_fused_docs]
        sources = [self._doc_source(doc, idx) for idx, doc in enumerate(all_docs, start=1)]
        #写缓存
        if self.cache and cache_key and all_docs:
            self.cache.set_json(
                cache_key,
                {"documents": [self._doc_to_cache(doc) for doc in all_docs]},
                expire=Config.RETRIEVAL_CACHE_TTL,
            )

        if subquery_stats:
            print("subquery retrieval summary:")
            for idx, (_, count) in enumerate(subquery_stats, start=1):
                print(f"  - query {idx}: {count} docs")
        print(f"final fused docs: {len(all_docs)}")
        #返回结果
        return {"documents": all_docs, "retrieval_sources": sources, "retrieval_cache_hit": False}

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
    "【回答要求】\n"
    "1. 引用具体条款时，注明来源文档名称（例如：根据《员工请假管理制度》第四条）。\n"
    "2. 如果检索到的资料不足以回答问题，请明确说“资料中未找到相关信息”。\n"
    "3. 回答应简洁、结构化，可使用分点或表格帮助理解。\n"
    "4. 对于涉及金额、天数、百分比等具体数字，务必核对准确。\n"
    "5. 禁止给出超出制度范围的建议（如“可以申请更多年假”）。\n"
    "6. 如果问题包含多个子问题，必须逐项覆盖，不能漏答。\n"
    "7. 若上下文中存在相互矛盾的信息，先说明冲突，再按优先级给出结论。\n"
    "8. 只允许复述文档中明确出现的规则，禁止补充文档未出现的条件、阈值或例外。\n"
)
        prompt = f"{system_prompt}\n\n根据资料：{context} 回答：{rewritten}"
        response = await self.llm.ainvoke(prompt) 
        return {"answer": response.content}
