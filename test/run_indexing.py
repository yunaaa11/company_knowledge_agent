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
    
    # 状态标记
    bm25_exists = os.path.exists(Config.bm25_path)
    new_docs_to_vector_db = []        # 只存放需要新添加到向量库的文档（未处理过的文件）
    all_docs_for_indexing = []      # 用于构建 BM25 的全量文档
    
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
                all_docs_for_indexing.extend(docs)
                    
                if not already_processed:
                        new_docs_to_vector_db.extend(docs)
                        cache.update_cache(file_hash)     # 记录已处理
                        print(f"✅新增/修改文件已加载: {file}")
                else:
                        print(f"跳过已处理文件（内容未变）: {file}")
            except Exception as e:
                        print(f"❌解析失败 {file}: {e}")

    # 3. 核心修复逻辑：写入向量库 (Chroma)
    if new_docs_to_vector_db:
        print(f"正在将 {len(new_docs_to_vector_db)} 个新文档存入向量库...")
        parent_retriever.add_documents(new_docs_to_vector_db)
    else:
        print("💡 向量库无需更新。")

    # 4. 核心修复逻辑：重建 BM25 索引
    # 触发条件：有新文档 OR 索引文件丢失
    if all_docs_for_indexing and (new_docs_to_vector_db or not bm25_exists):
        print(f"正在构建 BM25 索引 (文件总数: {len(all_docs_for_indexing)})...")
        
        # BM25 需要先切分
        child_chunks = vm.child_splitter.split_documents(all_docs_for_indexing)
        bm25_retriever = BM25Retriever.from_documents(child_chunks)
        
        # 确保目录存在并保存
        os.makedirs(os.path.dirname(Config.bm25_path), exist_ok=True)
        with open(Config.bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print(f"✅ BM25 索引已成功保存至: {Config.bm25_path}")
    elif not all_docs_for_indexing:
        print("❌ 错误：未发现任何有效文档。")
    else:
        print("💡 BM25 索引文件已存在且无新文档，跳过重建。")

if __name__ == "__main__":
    run_indexing()