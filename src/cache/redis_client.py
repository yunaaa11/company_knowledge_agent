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
    def generate_query_key(self,query:str):
        """用md5把query生成缓存"""
        return f"rag_cache:{hashlib.md5(query.encode()).hexdigest()}"
