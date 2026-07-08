import os
from dotenv import load_dotenv
load_dotenv()



def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default

def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

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
    HUGGINGFACEHUB_MODEL_NAME=os.getenv("HUGGINGFACEHUB_MODEL_NAME") 
    # Redis 配置
    REDIS_HOST = os.getenv("REDIS_HOST", "redis-server")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
    # 缓存开关
    ENABLE_CACHE = _get_bool("ENABLE_CACHE", True)
    # 缓存版本：用于文档重建索引或提示词变更后自动失效旧缓存
    INDEX_VERSION = os.getenv("INDEX_VERSION", "v1")
    PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
    CACHE_KEY_PREFIX = os.getenv("CACHE_KEY_PREFIX", "rag_cache")
    # Retrieval and rerank tuning
    VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 12)
    BM25_TOP_K = _get_int("BM25_TOP_K", 12)
    RERANK_TOP_N = _get_int("RERANK_TOP_N", 8)
    RERANK_MIN_SCORE = _get_float("RERANK_MIN_SCORE", 0.2)
    RERANK_SCORE_DROP = _get_float("RERANK_SCORE_DROP", 0.25)
    MAX_FUSED_DOCS = _get_int("MAX_FUSED_DOCS", 6)

    # Layered cache TTLs, in seconds
    REWRITE_CACHE_TTL = _get_int("REWRITE_CACHE_TTL", 3600)
    RETRIEVAL_CACHE_TTL = _get_int("RETRIEVAL_CACHE_TTL", 1800)
    ANSWER_CACHE_TTL = _get_int("ANSWER_CACHE_TTL", 3600)

# Parent-child chunking. Parent chunks feed the LLM; child chunks improve precise retrieval.
Config.PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1000)
Config.PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 120)
Config.CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 300)
Config.CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)

# Retrieval quality features
Config.ENABLE_INTENT_FILTER = _get_bool("ENABLE_INTENT_FILTER", True)
Config.ENABLE_HYDE = _get_bool("ENABLE_HYDE", False)
Config.HYDE_CACHE_TTL = _get_int("HYDE_CACHE_TTL", 3600)
Config.ORIGINAL_QUERY_WEIGHT = _get_float("ORIGINAL_QUERY_WEIGHT", 1.5)
Config.REWRITTEN_QUERY_WEIGHT = _get_float("REWRITTEN_QUERY_WEIGHT", 0.8)
Config.HYDE_QUERY_WEIGHT = _get_float("HYDE_QUERY_WEIGHT", 0.4)
Config.SOURCE_RULE_BOOST = _get_float("SOURCE_RULE_BOOST", 0.18)
