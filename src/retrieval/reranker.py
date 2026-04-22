import hashlib

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank # 或使用 BGE 模型
from flashrank import Ranker
class RerankProcessor:
    def __init__(self, base_retriever,top_n=10):
        # 1. 初始化压缩器 (Reranker)
        self.top_n = top_n
        self.compressor = FlashrankRerank(top_n=self.top_n)
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=base_retriever
        )
    
    def _generate_fingerprint(self, doc):
         source = doc.metadata.get("source", "")
         content_preview = doc.page_content[:200]
         return hashlib.md5(f"{source}_{content_preview}".encode("utf-8")).hexdigest()
    
    def retrieve(self, query: str, threshold=0.1):
        """
        检索、重排、去重，过滤掉分数低于阈值的文档
        """
        try:
            print("=== 混合检索原始结果（Rerank之前）===")
            raw = self.compression_retriever.base_retriever.invoke(query)
            for i, doc in enumerate(raw):
             print(f"  [{i}] 来源: {doc.metadata.get('source')}")
             print(f"       内容: {doc.page_content[:60]}")
            print(f"=== 共 {len(raw)} 条 ===")
            # 1. 调用压缩检索器（基础检索 + 重排序）
            results=self.compression_retriever.invoke(query) 
            print(f"DEBUG: 原始检索返回了 {len(results)} 条结果")  
            for i, doc in enumerate(results):
                score = doc.metadata.get("relevance_score", 0.0)
                source = doc.metadata.get("source", "未知")
                print(f"  - 排名 {i+1}: [分数 {score:.4f}] 来源: {source}")  
            # 2. 去重：基于文档内容的前 200 个字符（避免相邻 chunk 内容重复）
            seen = set()
            unique_results = []
            for doc in results:
                fingerprint = self._generate_fingerprint(doc)
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
            
            if not filtered_results:
                print("⚠️ 阈值过滤后结果为空，退回未过滤结果")
                return unique_results  # 至少保留去重后的结果
            return filtered_results
        
        except Exception as e:
            print(f"Rerank Error: {e}")
            # 降级：直接返回基础混合检索器的原始结果（未重排、未去重、未过滤）
            try:
                raw_results = self.compression_retriever.base_retriever.invoke(query)
                return raw_results[:self.top_n]
            except:
                return []