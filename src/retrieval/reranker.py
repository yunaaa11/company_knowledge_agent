import hashlib
import os

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from config import Config
from src.retrieval.intent import classify_query_intent


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
        """递归找出所有底层检索器"""
        yield retriever
        for child in getattr(retriever, "retrievers", []) or []:
            yield from self._iter_retrievers(child)

    def _apply_vector_filter(self, intent):
        """瞬态过滤器 动态修改向量库的 search_kwargs，只检索对应类别的文档（比如问“年假”就只查 policy_type="leave" 的索引）。"""
        previous = []
        if not intent or not Config.ENABLE_INTENT_FILTER:
            return previous
        for retriever in self._iter_retrievers(self.compression_retriever.base_retriever):
            if hasattr(retriever, "search_kwargs"):
                # 深拷贝一份旧的 search_kwargs（防止后续修改影响原对象
                old_kwargs = dict(getattr(retriever, "search_kwargs", {}) or {})
                new_kwargs = dict(old_kwargs)
                new_kwargs["filter"] = {"policy_type": intent}
                retriever.search_kwargs = new_kwargs
                previous.append((retriever, old_kwargs))
        return previous

    @staticmethod
    def _restore_vector_filter(previous):
        """瞬态过滤器"""
        for retriever, old_kwargs in previous:
            retriever.search_kwargs = old_kwargs

    @staticmethod
    def _filter_by_intent(docs, intent, min_keep=2):
        """后过滤兜底"""
        #即使向量检索时漏过了意图过滤，这里再强制筛一遍。但有个保底逻辑：如果筛完后剩下的文档少于 2 个，说明过滤太狠了，干脆全部保留，宁滥勿缺
        if not intent or not Config.ENABLE_INTENT_FILTER:
            return docs
        matched = [doc for doc in docs if (doc.metadata or {}).get("policy_type") == intent]
        return matched if len(matched) >= min_keep else docs

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

    def retrieve(self, query: str):
        raw = []
        #意图识别 + 预过滤
        intent = classify_query_intent(query)
        previous_filters = self._apply_vector_filter(intent)
        try:
            #初步召回（Base Retriever）
            raw = self.compression_retriever.base_retriever.invoke(query)
            raw = self._filter_by_intent(raw, intent)
            print(f"retrieval summary: query='{query}' | raw={len(raw)} | intent={intent}")

            results = self.compression_retriever.invoke(query)
            results = self._filter_by_intent(results, intent)
            print(f"rerank summary: query='{query}' | reranked={len(results)}")

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
            #分数断崖截断算法
            for i in range(1, len(unique_results)):
                current_score = scores[i]
                prev_score = scores[i - 1]
                if (prev_score - current_score) > self.score_drop_threshold:
                    break
                if current_score < self.min_score:
                    break
                filtered_results.append(unique_results[i])
            #保底兜底
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
            print(f"Rerank error: {e}")
            return raw[: self.top_n]
        finally:
        #无论重排序成功、失败、抛异常，只要 previous_filters 不为空，就一定会在最后一步把检索器恢复原状。
            self._restore_vector_filter(previous_filters)
