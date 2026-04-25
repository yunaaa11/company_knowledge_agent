import os
from dotenv import load_dotenv
load_dotenv()
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")
    chunk_size = int(os.getenv("chunk_size", 500))
    chunk_overlap = int(os.getenv("chunk_overlap", 50))
    separators=os.getenv("separators")
    db_path= os.getenv("db_path")
    store_path= os.getenv("store_path")
    bm25_path= os.getenv("bm25_path") 
    cache_file= os.getenv("cache_file") 
    HF_TOKEN= os.getenv("HF_TOKEN")   
    HF_ENDPOINT= os.getenv("HF_ENDPOINT") 
    # Redis 配置
    REDIS_HOST = os.getenv("REDIS_HOST", "redis-server")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
    # 缓存开关
    ENABLE_CACHE = os.getenv("ENABLE_CACHE") 