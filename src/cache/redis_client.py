import hashlib
import json
import os
from pathlib import Path
from config import Config

try:
    import redis
except ImportError:
    redis = None


class RedisCache:
    def __init__(self):
        self.client = None
        self.local_cache_dir = Path(os.getenv("LOCAL_CACHE_DIR", ".cache/rag_cache"))
        self.expire = getattr(Config, "ANSWER_CACHE_TTL", 3600)
        if redis is None:
            print("Redis package is not installed; cache is disabled.")
            return
        try:
            self.client = redis.Redis(
                host=getattr(Config, "REDIS_HOST", "localhost"),
                port=getattr(Config, "REDIS_PORT", 6379),
                password=getattr(Config, "REDIS_PASSWORD", None),
                db=0,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )
            self.client.ping()
        except Exception as e:
            print(f"Redis unavailable, using local cache fallback: {e}")
            self.client = None

    @staticmethod
    def _json_default(value):
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    def _local_path(self, key: str) -> Path:
        digest = hashlib.md5(key.encode()).hexdigest()
        return self.local_cache_dir / f"{digest}.json"

    def _local_get(self, key: str):
        path = self._local_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Local cache read error: {e}")
            return None

    def _local_set(self, key: str, value: dict):
        try:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            self._local_path(key).write_text(
                json.dumps(value, ensure_ascii=False, default=self._json_default),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Local cache write error: {e}")

    def get_cache(self, key: str):
        """
        根据 key 获取缓存数据
        参数:
            key: 缓存键
        返回:
            dict | None: 反序列化后的 JSON 对象，若 key 不存在或发生异常则返回 None
        """
        try:
            if not self.client:
                return self._local_get(key)
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
        if not self.client:
            self._local_set(key, value)
            return
        self.client.setex(
            key,
            self.expire,
            json.dumps(value, ensure_ascii=False, default=self._json_default),
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

    def get_json(self, key: str):
        return self.get_cache(key)

    def set_json(self, key: str, value: dict, expire: int | None = None):
        ttl = expire or self.expire
        try:
            if not self.client:
                self._local_set(key, value)
                return
            self.client.setex(
                key,
                ttl,
                json.dumps(value, ensure_ascii=False, default=self._json_default),
            )
        except Exception as e:
            print(f"Redis Error: {e}")

    def generate_stage_key(
        self,
        stage: str,
        query: str,
        chat_history=None,
        index_version: str = "v1",
        prompt_version: str = "v1",
        prefix: str = "rag_cache",
    ):
        history_text = json.dumps(chat_history or [], ensure_ascii=False, sort_keys=True)
        raw = f"stage={stage}|query={query}|history={history_text}|idx={index_version}|prompt={prompt_version}"
        return f"{prefix}:{stage}:{hashlib.md5(raw.encode()).hexdigest()}"
