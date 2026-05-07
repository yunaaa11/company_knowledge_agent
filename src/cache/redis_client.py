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
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            print(f"Redis Error: {e}")
            return None

    def set_cache(self, key: str, value: dict):
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
