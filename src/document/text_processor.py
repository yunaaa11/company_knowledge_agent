from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config

DEFAULT_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", "?", "?", ";", ".", " ", ""]


class DocumentSplitter:
    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.PARENT_CHUNK_SIZE,
            chunk_overlap=Config.PARENT_CHUNK_OVERLAP,
            separators=DEFAULT_SEPARATORS,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHILD_CHUNK_SIZE,
            chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
            separators=DEFAULT_SEPARATORS,
        )

    def split_parent(self, documents):
        return self.parent_splitter.split_documents(documents)

    def split_child(self, documents):
        return self.child_splitter.split_documents(documents)

    def split(self, documents):
        return self.split_parent(documents)
