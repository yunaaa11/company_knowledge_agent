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
    
    def retrieve(self, query: str):
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
            if not unique_results:
                print("⚠️ 去重后结果为空，退回原始结果")
                return results  # 至少保留重排后的结果

             # 3. 动态裁剪逻辑：基于得分梯度（断层检测）
            scores = [doc.metadata.get("relevance_score", 0.0) for doc in unique_results]

            # 初始加入第一个最高分的文档
            filtered_results=[unique_results[0]]
            print(f"--- 原始召回: {len(unique_results)} 条 | 最高分: {scores[0]:.4f} ---")

            for i in range(1,len(unique_results)):
                current_score = scores[i] #i表示去重后的索引
                prev_score=scores[i-1]
                # 策略 A: 梯度截断（断层检测）
                # 如果当前文档比前一个文档分数骤降超过 0.3，认为后面全是噪声
                if(prev_score - current_score)>0.3:
                    print(f"触发梯度截断：分差 {prev_score-current_score:.2f} ")
                    break
                # 策略 B: 保底阈值
                # 即便没有明显断层，分数过低（如低于 0.15）也不予采用
                if current_score<0.15:
                    print(f"触发保底过滤：分数 {current_score:.4f} < 0.15")
                    break
                filtered_results.append(unique_results[i])
            # 4. 返回最终结果，受 top_n 限制
            final_docs=filtered_results[:self.top_n]
            print(f"✅ 动态裁剪完成，最终保留: {len(final_docs)} 条")

            for i,doc in enumerate(final_docs):
                s=doc.metadata.get("relevance_score",0.0)
                print(f"  [{i}] 分数: {s:.4f} | 来源: {doc.metadata.get('source')}")
            return final_docs
        except Exception as e:
            print(f"❌ Rerank 过程出错: {e}")
            # 降级：如果出错，尝试返回未去重的原始检索结果
            return []