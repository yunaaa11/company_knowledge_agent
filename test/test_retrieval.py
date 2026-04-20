import sys
import os
# 1. 将项目根目录加入环境路径
# os.path.abspath(__file__) 获取当前文件的绝对路径
# 第一层 dirname 是 test/ 目录，第二层 dirname 是 BUSSINESS/ 根目录
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import RerankProcessor

def run_test():
    # 1. 初始化
    vm = VectorStoreManager()
    hs = HybridSearcher(vm)
    
    # 2. 组装：混合检索器 -> 重排序器
    base_retriever = hs.get_ensemble_retriever()
    reranker = RerankProcessor(base_retriever)
    
    # 3. 执行
    query = "绩效等级S级需要多少分？"
    # 检索 + 重排序，返回 top_n 个父文档
    results = reranker.retrieve(query)
    
    for i, doc in enumerate(results):
        print(f"\n[结果 {i+1}] 来源: {doc.metadata.get('source', '未知')}")
        print(f"内容预览: {doc.page_content[:500]}...")

if __name__ == "__main__":
    run_test()