import hashlib
import os

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from config import Config
from src.retrieval.intent import classify_query_intent, classify_query_intents


class RerankProcessor:
    def __init__(self, base_retriever, top_n=None, min_score=None, score_drop_threshold=None, verbose=None, preview_limit=3):
        self.top_n = top_n if top_n is not None else Config.RERANK_TOP_N
        self.min_score = min_score if min_score is not None else Config.RERANK_MIN_SCORE
        self.score_drop_threshold = score_drop_threshold if score_drop_threshold is not None else Config.RERANK_SCORE_DROP
        self.verbose = os.getenv("RETRIEVAL_VERBOSE", "false").lower() == "true" if verbose is None else verbose
        self.preview_limit = preview_limit
        self.compressor = FlashrankRerank(top_n=self.top_n)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever,
        )

    def _iter_retrievers(self, retriever):
        yield retriever
        for child in getattr(retriever, "retrievers", []) or []:
            yield from self._iter_retrievers(child)

    def _apply_vector_filter(self, intent):
        previous = []
        if not intent or not Config.ENABLE_INTENT_FILTER:
            return previous
        if isinstance(intent, (list, tuple, set)):
            intent_values = [item for item in intent if item]
            if len(intent_values) != 1:
                return previous
            intent = intent_values[0]
        for retriever in self._iter_retrievers(self.compression_retriever.base_retriever):
            if hasattr(retriever, "search_kwargs"):
                old_kwargs = dict(getattr(retriever, "search_kwargs", {}) or {})
                new_kwargs = dict(old_kwargs)
                new_kwargs["filter"] = {"policy_type": intent}
                retriever.search_kwargs = new_kwargs
                previous.append((retriever, old_kwargs))
        return previous

    @staticmethod
    def _restore_vector_filter(previous):
        for retriever, old_kwargs in previous:
            retriever.search_kwargs = old_kwargs

    @staticmethod
    def _filter_by_intent(docs, intent, min_keep=2):
        if not intent or not Config.ENABLE_INTENT_FILTER:
            return docs
        intents = list(intent) if isinstance(intent, (list, tuple, set)) else [intent]
        intents = [item for item in intents if item]
        if not intents:
            return docs

        min_required = 1 if len(intents) == 1 else min_keep

        # Step 1: 严格匹配 — policy_type == intent
        strict_matched = [doc for doc in docs if (doc.metadata or {}).get("policy_type") in intents]
        if len(strict_matched) >= min_required:
            return strict_matched

        # Step 2: 宽松多标签匹配 — intent 在 policy_types (列表) 中
        strict_ids = {id(d) for d in strict_matched}
        relaxed = list(strict_matched)
        for doc in docs:
            if id(doc) in strict_ids:
                continue
            policy_types = (doc.metadata or {}).get("policy_types", [])
            if any(intent_item in policy_types for intent_item in intents):
                relaxed.append(doc)

        if len(relaxed) >= min_required:
            return relaxed

        # 完全回退 —— 返回所有文档
        return docs

    @staticmethod
    def _score(doc):
        score = doc.metadata.get("relevance_score", 0.0)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _fingerprint(doc):
        source = doc.metadata.get("source", "")
        content_preview = doc.page_content[:200]
        return hashlib.md5(f"{source}_{content_preview}".encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_log_text(value):
        return str(value).encode("gbk", errors="backslashreplace").decode("gbk")

    @staticmethod
    def _source_rule_score(query, source):
        text = query or ""
        source = source or ""
        rules = [
            (("离职", "交接"), "11_员工入职转正与离职交接制度", 2),
            (("账号", "账户", "权限", "设备"), "11_员工入职转正与离职交接制度", 1),
            (("预付款", "付款", "预算"), "12_预算管理与付款审批制度", 2),
            (("合同", "采购", "法务", "审核"), "07_采购与合同审批制度", 2),
            (("密码", "数据级别", "敏感数据"), "04_信息安全与数据分级管理规范", 2),
            (("客户资料", "商业秘密"), "13_客户资料与商业秘密保护制度", 2),
            (("工单", "P1", "故障", "响应", "解决"), "03_IT服务与设备权限管理制度", 2),
            (("绩效", "申诉", "考核"), "06_绩效考核与申诉管理制度", 2),
        ]
        score = 0
        for keywords, source_marker, weight in rules:
            if source_marker in source and any(keyword in text for keyword in keywords):
                score += weight
        return score

    def _apply_source_boost(self, query, docs):
        for doc in docs:
            metadata = doc.metadata or {}
            base_score = self._score(doc)
            boost = self._source_rule_score(query, metadata.get("source", "")) * Config.SOURCE_RULE_BOOST
            if boost:
                metadata["relevance_score"] = base_score + boost
                metadata["source_rule_boost"] = boost
                doc.metadata = metadata
        docs.sort(key=self._score, reverse=True)
        return docs

    def retrieve(self, query: str, prev_intent: str | None = None):
        raw = []
        intent = classify_query_intent(query, prev_intent=prev_intent)
        intents = classify_query_intents(query, prev_intent=prev_intent)
        previous_filters = self._apply_vector_filter(intents or intent)
        try:
            safe_query = self._safe_log_text(query)
            print(f"retrieval summary: query='{safe_query}' | intent={intent} | intents={intents}")

            results = self.compression_retriever.invoke(query)
            results = self._filter_by_intent(results, intents or intent)
            results = self._apply_source_boost(query, results)
            print(f"rerank summary: query='{safe_query}' | reranked={len(results)}")

            seen = set()
            unique_results = []
            for doc in results:
                fingerprint = self._fingerprint(doc)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                unique_results.append(doc)

            if not unique_results:
                return raw[: self.top_n]

            scores = [self._score(doc) for doc in unique_results]
            filtered_results = [unique_results[0]]
            for i in range(1, len(unique_results)):
                current_score = scores[i]
                prev_score = scores[i - 1]
                if (prev_score - current_score) > self.score_drop_threshold:
                    break
                if current_score < self.min_score:
                    break
                filtered_results.append(unique_results[i])

            if len(filtered_results) < 2:
                for doc in unique_results[1:]:
                    if doc in filtered_results:
                        continue
                    if self._score(doc) >= self.min_score * 0.8:
                        filtered_results.append(doc)
                    if len(filtered_results) >= min(2, len(unique_results)):
                        break

            final_docs = filtered_results[: self.top_n]
            for i, doc in enumerate(final_docs, start=1):
                doc.metadata.setdefault("rerank_rank", i)
            return final_docs
        except Exception as e:
            print(f"Rerank error: {self._safe_log_text(e)}")
            return raw[: self.top_n]
        finally:
            self._restore_vector_filter(previous_filters)
