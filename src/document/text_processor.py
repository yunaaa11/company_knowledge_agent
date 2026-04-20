from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config
class DocumentSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        # 父文档切分（较大，保留上下文）
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.chunk_size,
            chunk_overlap=Config.chunk_overlap,
            separators=Config.separators
        )
        # 子文档切分（较小，用于向量检索定位）
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.chunk_size // 4,
            chunk_overlap=Config.chunk_overlap // 2
        )

    def split(self, documents):
        # 新的列表，其中每个元素是切分后的小 Document 块
        return self.parent_splitter.split_documents(documents)