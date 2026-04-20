import os
import sys
import pickle
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录 (即 test 的上一级)
root_dir = os.path.dirname(current_dir)
# 将根目录添加到 Python 搜索路径
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.retrieval.vector_store import VectorStoreManager
from src.document.doc_loader import DocumentParser
from src.document.cache import DocCacheManager 
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from config import Config

def run_indexing():
    # 1. 初始化组件
    vm = VectorStoreManager()
    # 按照你的结构，这里应该调用你定义的 ParentDocumentRetriever
    parent_retriever = vm.get_parent_retriever() 
    cache = DocCacheManager()
    raw_data_path = "data/raw/"

    new_docs = []          # 只存放需要新添加到向量库的文档（未处理过的文件）
    all_docs_for_bm25 = [] # 存放所有文档（用于 BM25 重建，无论新旧）
    
    # 手动遍历文件夹，调用你的解析器
    print(f"正在从 {raw_data_path} 加载并解析文档...")
    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            if not file.endswith(('.md', '.txt', '.pdf', '.docx')):
                continue
            file_path = os.path.join(root, file)
            file_hash = cache.get_file_hash(file_path)
            already_processed = cache.is_processed(file_path)
            try:
                # 使用写的 DocumentParser，它能处理不同格式
                docs = DocumentParser.parse(file_path)
                all_docs_for_bm25.extend(docs)
                    
                if not already_processed:
                        new_docs.extend(docs)
                        cache.update_cache(file_hash)     # 记录已处理
                        print(f"✅新增/修改文件已加载: {file}")
                else:
                        print(f"⏭️跳过已处理文件（内容未变）: {file}")
            except Exception as e:
                        print(f"❌解析失败 {file}: {e}")

    # 3. 重建 BM25 索引（仅当有新文档时，避免无意义重建）
    if new_docs and all_docs_for_bm25:
        print("正在全量重建 BM25 索引...")
        child_chunks = vm.child_splitter.split_documents(all_docs_for_bm25)
        bm25_retriever = BM25Retriever.from_documents(child_chunks)
        os.makedirs(os.path.dirname(Config.bm25_path), exist_ok=True)
        with open(Config.bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print("BM25 索引已更新。")
    elif not new_docs:
        print("没有新文档，BM25 索引无需重建。")
    else:
        print("警告：没有解析到任何文档，跳过 BM25 构建。")

if __name__ == "__main__":
    run_indexing()