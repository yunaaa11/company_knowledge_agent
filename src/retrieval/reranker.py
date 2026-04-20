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

    def retrieve(self, query: str):
        try:
            return self.compression_retriever.invoke(query)
        except Exception as e:
            print(f"Rerank Error: {e}")
            # 发生错误时降级返回混合检索原始结果
            return self.compression_retriever.base_retriever.invoke(query)