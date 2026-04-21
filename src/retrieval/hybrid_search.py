import pickle
from langchain_classic.retrievers import EnsembleRetriever
from config import Config
import os

class HybridSearcher:
    def __init__(self, vector_manager, bm25_path=Config.bm25_path):
        self.vector_retriever = vector_manager.get_parent_retriever()
    
        # 加载 BM25 (注意添加路径检查)
        #bm25_path:保存 BM25 检索器的索引数据（倒排索引、文档长度等统计信息）
        if os.path.exists(Config.bm25_path):
            # 加载索引构建阶段生成的 BM25 
            with open(Config.bm25_path, "rb") as f:
                 # pickle.load 反序列化，将文件中的对象恢复为 BM25 检索器实例
                self.bm25_retriever = pickle.load(f)
            # 设置每次检索返回的文档数量
            self.bm25_retriever.k = 10
        else:
                print(f"⚠️ 警告: 未找到 BM25 索引文件 {Config.bm25_path}")
                self.bm25_retriever = None
    
    def get_ensemble_retriever(self):
        """
        融合 RRF。
        注意：如果 bm25 加载失败，退化为单一向量检索
        """
        if not self.bm25_retriever:
            return self.vector_retriever
            
        return EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            #微调权重：提升 BM25 的影响力，利用关键词频次锁定行政文档
            weights=[0.6, 0.4] 
        )