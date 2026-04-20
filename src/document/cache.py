import hashlib
from config import Config
class DocCacheManager:
    def __init__(self, cache_file=Config.cache_file):
        self.cache_file = cache_file
        #存储所有已处理文件的 MD5 哈希值（字符串形式
        self.processed_hashes = self._load_cache()
    
    def _load_cache(self):
        """加载缓存"""
        try:
            #只读方式打开缓存文件，逐行读取
            with open(self.cache_file, "r") as f:
                return set(line.strip() for line in f)
        except FileNotFoundError:
            return set()

    def get_file_hash(self, file_path):
        """计算文件 MD5"""
        hash_md5 = hashlib.md5()
        #以二进制方式打开文件（"rb"），分块（每块 4096 字节）读取，
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        #逐步更新 MD5 计算对象，最终返回 32 位十六进制字符串作为文件的哈希值
        return hash_md5.hexdigest()

    def is_processed(self, file_path):
        """检查文件是否已处理"""
        #先计算目标文件的 MD5，然后判断该哈希值是否已经存在于缓存集合中
        file_hash = self.get_file_hash(file_path)
        return file_hash in self.processed_hashes

    def update_cache(self, file_hash):
        """将新处理的文件哈希追加到缓存文件中"""
        with open(self.cache_file, "a") as f:
            f.write(f"{file_hash}\n")
        #同时更新内存中的集合
        self.processed_hashes.add(file_hash)