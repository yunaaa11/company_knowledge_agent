from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
os.environ["HF_ENDPOINT"] = Config.HF_ENDPOINT
os.environ["HF_TOKEN"] =Config.HF_TOKEN
class VectorStoreManager:
    def __init__(self, db_path=Config.db_path, store_path=Config.store_path):
        self.db_path = Config.db_path
        self.store_path = Config.store_path
        os.makedirs(store_path, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name= Config.HUGGINGFACEHUB_MODEL_NAME)
        self.vectorstore = Chroma(
            collection_name="enterprise_paper",
            embedding_function=self.embeddings,
            persist_directory=db_path
        )
         # 1. 原始文件存储（bytes级别）
        raw_fs = LocalFileStore(store_path)
        # 2. 包装成符合 Docstore 接口的对象
        self.docstore = create_kv_docstore(raw_fs)

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.chunk_size, 
            chunk_overlap=Config.chunk_overlap
        )

    def get_parent_retriever(self):
        """
        检索器:用小查大、自动反向查找父文档
        """
        return ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
        ) 