import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config

os.environ["HF_ENDPOINT"] = Config.HF_ENDPOINT or ""
os.environ["HF_TOKEN"] = Config.HF_TOKEN or ""

SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", "?", "?", ";", ".", " ", ""]


class VectorStoreManager:
    def __init__(self, db_path=Config.db_path, store_path=Config.store_path):
        self.db_path = db_path or Config.db_path
        self.store_path = store_path or Config.store_path
        os.makedirs(self.store_path, exist_ok=True)
       
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.HUGGINGFACEHUB_MODEL_NAME)
        self.vectorstore = Chroma(
            collection_name="enterprise_paper",
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )
       
        raw_fs = LocalFileStore(self.store_path)
        self.docstore = create_kv_docstore(raw_fs)
      
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.PARENT_CHUNK_SIZE,
            chunk_overlap=Config.PARENT_CHUNK_OVERLAP,
            separators=SEPARATORS,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHILD_CHUNK_SIZE,
            chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
            separators=SEPARATORS,
        )

    def get_parent_retriever(self):
        return ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            parent_splitter=self.parent_splitter,
            child_splitter=self.child_splitter,
        )

