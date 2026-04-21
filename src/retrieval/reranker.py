from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank # 或使用 BGE 模型
from flashrank import Ranker
class RerankProcessor:
    def __init__(self, base_retriever):
        # 1. 初始化压缩器 (Reranker)
        self.compressor = FlashrankRerank(top_n=3)
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=base_retriever
        )

    def retrieve(self, query: str, threshold=0.5):
        """
        检索、重排、去重，过滤掉分数低于阈值的文档
        """
        try:
            # 1. 调用压缩检索器（基础检索 + 重排序）
            results=self.compression_retriever.invoke(query)
            # 2. 去重：基于文档内容的前 200 个字符（避免相邻 chunk 内容重复）
            seen = set()
            unique_results = []
            for doc in results:
                fingerprint = doc.page_content[:200]   # 简单指纹，可改为更可靠的 doc_id
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    unique_results.append(doc)

             # 3. 阈值过滤：只保留重排分数 >= threshold 的文档
            filtered_results = []
            for doc in unique_results:
                #重排后的分数通常存储在 metadata['relevance_score'] 中
                # 获取分数，如果没有分数则默认为 0.0
                score = doc.metadata.get("relevance_score", 0.0)
                
                if score >= threshold:
                    filtered_results.append(doc)
                else:
                    source = doc.metadata.get('source', 'unknown')
                    print(f"--- 过滤噪声文档 (分数: {score:.4f} < {threshold}): {doc.metadata.get('source')} ---")
            
            # 如果过滤后一个都没剩，可以考虑降级处理或返回空
            return filtered_results
        
        except Exception as e:
            print(f"Rerank Error: {e}")
            # 降级：直接返回基础混合检索器的原始结果（未重排、未去重、未过滤）
            return self.compression_retriever.base_retriever.invoke(query)