import os
import pickle
import shutil
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
from langchain_community.retrievers import BM25Retriever
from config import Config
from src.document.doc_loader import DocumentParser
from src.document.metadata import prepare_documents_for_indexing
from src.retrieval.vector_store import VectorStoreManager
SUPPORTED_EXTS = (".md", ".txt", ".pdf", ".docx")

#把磁盘上的原始文档（PDF/Markdown/Word）加工、切分、向量化，并分别存入向量库（Chroma）和关键词检索库（BM25），为后续的在线检索提供数据底座。

def reset_index_storage():
    """删除 Chroma 持久化目录、Docstore 文件存储、BM25 序列化文件、缓存文件"""
    for path in [Config.db_path, Config.store_path]:
        if path and os.path.exists(path):
            shutil.rmtree(path)
    if Config.bm25_path and os.path.exists(Config.bm25_path):
        os.remove(Config.bm25_path)
    if Config.cache_file and os.path.exists(Config.cache_file):
        os.remove(Config.cache_file)


def load_documents(raw_data_path="data/raw"):
    """加载与解析文档 单个文件解析失败不会中断整个流程"""
    all_docs = []
    for root, _, files in os.walk(raw_data_path):
        for file in files:
            if not file.lower().endswith(SUPPORTED_EXTS):
                continue
            file_path = os.path.join(root, file)
            try:
                docs = DocumentParser.parse(file_path)
                prepared = prepare_documents_for_indexing(docs)
                all_docs.extend(prepared)
                print(f"loaded {file_path}: {len(prepared)} section docs")
            except Exception as exc:
                print(f"failed to parse {file_path}: {exc}")
    return all_docs


def run_indexing(force_rebuild=True):
    #重建 保证数据一致性与可复现性
    if force_rebuild:
        reset_index_storage()

    documents = load_documents("data/raw")
    if not documents:
        print("No valid documents found.")
        return

    vm = VectorStoreManager()
    parent_retriever = vm.get_parent_retriever()

    print(f"adding {len(documents)} enriched parent documents to Chroma parent retriever...")
    parent_retriever.add_documents(documents)
    #保证 BM25 的索引粒度与 Chroma 的索引粒度完全一致
    child_chunks = vm.child_splitter.split_documents(documents)
    print(f"building BM25 index with {len(child_chunks)} child chunks...")
    # BM25 要用子块 保证了 BM25 的“高分辨率”匹配能力
    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = Config.BM25_TOP_K

    os.makedirs(os.path.dirname(Config.bm25_path), exist_ok=True)
    with open(Config.bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print("index rebuild complete")
    print(f"Chroma path: {Config.db_path}")
    print(f"Docstore path: {Config.store_path}")
    print(f"BM25 path: {Config.bm25_path}")
    print(f"Parent chunk: {Config.PARENT_CHUNK_SIZE}/{Config.PARENT_CHUNK_OVERLAP}")
    print(f"Child chunk: {Config.CHILD_CHUNK_SIZE}/{Config.CHILD_CHUNK_OVERLAP}")


if __name__ == "__main__":
    run_indexing(force_rebuild=True)
