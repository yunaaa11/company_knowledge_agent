import redis
import json
from config import Config
import hashlib
class RedisCache:
    def __init__(self):
        self.client=redis.Redis(
            host=getattr(Config, "REDIS_HOST", "localhost"),
            port=getattr(Config, "REDIS_PORT", 6379),
            db=0,
            decode_responses=True
        )
        self.expire=3600 # 默认 1 小时过期
    def get_cache(self,key:str):
        """获取缓存"""
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            print(f"Redis Error: {e}") # 生产环境建议只打日志不中断
            return None
    def set_cache(self,key:str,value:dict):
        """设置缓存"""
        self.client.setex(
            key,
            self.expire,
            json.dumps(value,ensure_ascii=False)
        )
    def generate_query_key(
        self,
        query: str,
        index_version: str = "v1",
        prompt_version: str = "v1",
        prefix: str = "rag_cache",
    ):
        """缓存键包含 query + 版本号，避免索引/提示词升级后命中过期缓存"""
        raw = f"{query}|idx={index_version}|prompt={prompt_version}"
        return f"{prefix}:{hashlib.md5(raw.encode()).hexdigest()}"
