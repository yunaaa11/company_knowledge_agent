import hashlib
import json
import redis
from config import Config


class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=getattr(Config, "REDIS_HOST", "localhost"),
            port=getattr(Config, "REDIS_PORT", 6379),
            password=getattr(Config, "REDIS_PASSWORD", None),
            db=0,
            decode_responses=True,
        )
        self.expire = 3600

    def get_cache(self, key: str):
        """
        根据 key 获取缓存数据
        参数:
            key: 缓存键
        返回:
            dict | None: 反序列化后的 JSON 对象，若 key 不存在或发生异常则返回 None
        """
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            print(f"Redis Error: {e}")
            return None

    def set_cache(self, key: str, value: dict):
        """
        将数据存入 Redis 缓存，自动设置过期时间
        参数:
            key:   缓存键
            value: 要缓存的字典对象（会序列化为 JSON 字符串）
        """
        self.client.setex(
            key,
            self.expire,
            json.dumps(value, ensure_ascii=False),
        )

    def generate_query_key(
        self,
        query: str,
        chat_history=None,
        index_version: str = "v1",
        prompt_version: str = "v1",
        prefix: str = "rag_cache",
    ):
        """根据查询内容、对话历史、知识库版本、提示模板版本,生成一个唯一的缓存键"""
        history_text = json.dumps(chat_history or [], ensure_ascii=False, sort_keys=True)
        raw = f"{query}|history={history_text}|idx={index_version}|prompt={prompt_version}"
        #计算 MD5 并拼接前缀
        return f"{prefix}:{hashlib.md5(raw.encode()).hexdigest()}"
