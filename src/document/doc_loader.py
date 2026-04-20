from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredWordDocumentLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader
)

class DocumentParser:
    #静态方法，意味着无需实例化类
    @staticmethod
    def parse(file_path: str):
        # 获取文件扩展名
        ext = file_path.split('.')[-1].lower()
        
        if ext == 'pdf':
            loader = PyMuPDFLoader(file_path) # 速度快且对中文友好
        elif ext in ['doc', 'docx']:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == 'md':
            loader = TextLoader(file_path, encoding='utf-8')
        else:#如 .txt、.log 或无扩展名
            loader = TextLoader(file_path, encoding='utf-8')
        # 返回文档内容
        return loader.load()