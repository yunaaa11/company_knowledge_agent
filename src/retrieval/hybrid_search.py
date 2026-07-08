import os
import pickle

from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from src.retrieval.intent import infer_policy_type, infer_policy_types


class HybridSearcher:
    def __init__(self, vector_manager, bm25_path=Config.bm25_path):
        self.vector_retriever = vector_manager.get_parent_retriever()
        if hasattr(self.vector_retriever, "search_kwargs"):
            self.vector_retriever.search_kwargs = {"k": Config.VECTOR_TOP_K}

        self.bm25_path = bm25_path or Config.bm25_path
        self.bm25_retriever = self._load_bm25()

    def _load_bm25(self):
        if not self.bm25_path or not os.path.exists(self.bm25_path):
            print(f"Warning: BM25 index file not found: {self.bm25_path}")
            return None

        try:
            with open(self.bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
            bm25_retriever.k = Config.BM25_TOP_K
            doc_count = len(getattr(bm25_retriever, "docs", []))
            print(f"BM25 loaded, docs: {doc_count}")
            return bm25_retriever
        except Exception as exc:
            print(f"Warning: failed to load BM25 index; rebuilding lightweight BM25: {exc}")
            return self._build_bm25_from_raw_docs()

    def _build_bm25_from_raw_docs(self):
        raw_dir = os.getenv("RAW_ENHANCED_DIR", "data/raw/enhanced")
        if not os.path.isdir(raw_dir):
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHILD_CHUNK_SIZE,
            chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
        )
        docs = []
        for name in os.listdir(raw_dir):
            if not name.endswith(".md"):
                continue
            if name.upper() == "README.MD":
                continue
            path = os.path.join(raw_dir, name)
            try:
                text = open(path, "r", encoding="utf-8").read()
            except UnicodeDecodeError:
                text = open(path, "r", encoding="utf-8-sig").read()
            metadata = {
                "source": path,
                "policy_type": infer_policy_type(path),
                "policy_types": infer_policy_types(path),
            }
            docs.extend(splitter.split_documents([Document(page_content=text, metadata=metadata)]))

        if not docs:
            return None
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = Config.BM25_TOP_K
        print(f"BM25 rebuilt from raw docs, chunks: {len(docs)}")
        return bm25_retriever

    def get_ensemble_retriever(self):
        """Return hybrid retriever, or vector retriever when BM25 is unavailable."""
        if not self.bm25_retriever:
            return self.vector_retriever

        return EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.35, 0.65],
        )

