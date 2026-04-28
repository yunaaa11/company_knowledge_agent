import hashlib
import os

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank # 或使用 BGE 模型
class RerankProcessor:
    def __init__(self, base_retriever, top_n=8, min_score=0.2, score_drop_threshold=0.2, verbose=None, preview_limit=3):
        # 1. 初始化压缩器 (Reranker)
        self.top_n = top_n
        self.min_score = min_score
        self.score_drop_threshold = score_drop_threshold
        self.verbose = (
            os.getenv("RETRIEVAL_VERBOSE", "false").lower() == "true"
            if verbose is None else verbose
        )
        self.preview_limit = preview_limit
        self.compressor = FlashrankRerank(top_n=self.top_n)
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=base_retriever
        )
    
    def _generate_fingerprint(self, doc):
         """为文档生成唯一标识。基于 来源 + 内容前200字符 的MD5值，用于后续去重。
            不同子查询可能召回同一文档片段，或者相邻 chunk 内容高度重叠，需要去重。
          """
         source = doc.metadata.get("source", "")
         content_preview = doc.page_content[:200]
         return hashlib.md5(f"{source}_{content_preview}".encode("utf-8")).hexdigest()
    
    def retrieve(self, query: str):
        """
        检索、重排、去重，过滤掉分数低于阈值的文档
        """
        raw = []
        try:
            #基础召回 仅调用基础检索器（未重排），用于降级备选和日志对比
            raw = self.compression_retriever.base_retriever.invoke(query)
            print(f"检索摘要: query='{query}' | 基础召回 {len(raw)} 条")
            if self.verbose:
                print("=== 混合检索原始结果（Rerank之前）===")
                for i, doc in enumerate(raw[:self.preview_limit]):
                    print(f"  [{i}] 来源: {doc.metadata.get('source')}")
                    print(f"       内容: {doc.page_content[:60]}")
                if len(raw) > self.preview_limit:
                    print(f"  ... 其余 {len(raw) - self.preview_limit} 条已省略")
           
            # 重排检索 基础检索 → 重排序器 → 返回按新分数排序的文档列表
            results = self.compression_retriever.invoke(query)
            print(f"重排摘要: query='{query}' | 重排后 {len(results)} 条")
            if self.verbose:
                for i, doc in enumerate(results[:self.preview_limit]):
                    score = doc.metadata.get("relevance_score", 0.0)
                    source = doc.metadata.get("source", "未知")
                    print(f"  - 排名 {i+1}: [分数 {score:.4f}] 来源: {source}")
                if len(results) > self.preview_limit:
                    print(f"  ... 其余 {len(results) - self.preview_limit} 条已省略")

            # 去重：基于指纹去重，保留首次出现的文档（通常分数最高的那个）
            seen = set()
            unique_results = []
            for doc in results:
                fingerprint = self._generate_fingerprint(doc)
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    unique_results.append(doc)
            if not unique_results:
                print("⚠️ 去重后结果为空，退回原始结果")
                return results[:self.top_n]

            # 3. 动态裁剪逻辑：基于得分梯度和最低保真阈值
            scores = [doc.metadata.get("relevance_score", 0.0) for doc in unique_results]
            # 初始加入第一个最高分的文档
            filtered_results = [unique_results[0]]
            print(f"裁剪摘要: 去重后 {len(unique_results)} 条 | 最高分 {scores[0]:.4f}")

            for i in range(1, len(unique_results)):
                current_score = scores[i] #i表示去重后的索引
                prev_score = scores[i-1]
                # 策略A：梯度截断 —— 如果分数骤降超过阈值，则后面全部丢弃
                if (prev_score - current_score) > self.score_drop_threshold:
                    print(f"触发梯度截断：分差 {prev_score-current_score:.2f}")
                    break
                # 策略B：保底阈值 —— 如果当前分数低于 min_score，则后面全部丢弃
                if current_score < self.min_score:
                    print(f"触发保底过滤：分数 {current_score:.4f} < {self.min_score:.2f}")
                    break
                filtered_results.append(unique_results[i])

            # 保底补全:保证至少保留 2 条高相关文档，避免召回过窄导致答案不完整
            if len(filtered_results) < 2:
                for doc in unique_results[1:]:
                    if doc in filtered_results:
                        continue
                    score = doc.metadata.get("relevance_score", 0.0)
                    if score >= self.min_score * 0.8:
                        filtered_results.append(doc)
                    if len(filtered_results) >= min(2, len(unique_results)):
                        break
                    
            # 返回最终结果，受 top_n 限制
            final_docs = filtered_results[:self.top_n]
            print(f"✅ 动态裁剪完成，最终保留: {len(final_docs)} 条")

            for i, doc in enumerate(final_docs):
                s = doc.metadata.get("relevance_score", 0.0)
                print(f"  [{i}] 分数: {s:.4f} | 来源: {doc.metadata.get('source')}")
            return final_docs
        except Exception as e:
            print(f"❌ Rerank 过程出错: {e}")
            # 降级：如果重排失败，回退到基础检索结果，避免评估被异常污染
            return raw[:self.top_n]