import json
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import Config
from src.agent.workflow import create_graph
from src.cache.redis_client import RedisCache
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.intent import classify_query_intent
from src.retrieval.reranker import RerankProcessor
from src.retrieval.vector_store import VectorStoreManager

router = APIRouter()
#这些对象在模块加载时只创建一次，后续所有请求复用它们，避免重复加载模型、连接数据库等昂贵操作。
vm = VectorStoreManager()
hs = HybridSearcher(vm)
reranker = RerankProcessor(hs.get_ensemble_retriever())
redis_cache = RedisCache() if Config.ENABLE_CACHE else None
llm = ChatOpenAI(
    model=Config.LLM_MODEL,
    openai_api_key=Config.OPENAI_API_KEY,
    openai_api_base=Config.OPENAI_BASE_URL,
    temperature=0,
)
agent_app = create_graph(vm, reranker, llm)


def serialize_documents(documents, limit: int = 6):
    serialized = []
    for rank, doc in enumerate(documents[:limit], start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        page_content = getattr(doc, "page_content", "") or ""
        snippet = " ".join(page_content.split())[:300]
        score = metadata.get("relevance_score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        serialized.append(
            {
                "rank": rank,
                "source": metadata.get("source", "unknown"),
                "relevance_score": score,
                "snippet": snippet,
                "page_content": page_content,
            }
        )
    return serialized


class ChatRequest(BaseModel):
    query: str = Field(..., example="北京的报销标准是多少？")
    chat_history: Optional[List[dict]] = Field(default_factory=list)


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def stream_generator():
        # 从 chat_history 中提取上一轮的 intent，用于追问继承
        prev_intent = None
        if request.chat_history:
            for item in reversed(request.chat_history):
                if isinstance(item, dict) and item.get("role") == "user":
                    prev_intent = classify_query_intent(item.get("content", ""))
                    break

        inputs = {
            "query": request.query,
            "chat_history": request.chat_history,
            "loop_step": 0,
            "prev_intent": prev_intent,
            "prev_documents": [],
        }
        # 缓存键生成与命中处理
        cache_key = None
        stateless_cache_key = None
        if redis_cache:
            # 检查缓存 根据 query、历史、知识库版本等生成缓存键。
            cache_key = redis_cache.generate_query_key(
                query=request.query,
                chat_history=request.chat_history,
                index_version=Config.INDEX_VERSION,  # 知识库版本，变了则旧缓存失效
                prompt_version=Config.PROMPT_VERSION,
                prefix=Config.CACHE_KEY_PREFIX,
            )
            #如果缓存命中，直接流式返回缓存的改写问句和完整答案，并标记 cache_hit: true，结束。
            stateless_cache_key = redis_cache.generate_query_key(
                query=request.query,
                chat_history=[],
                index_version=Config.INDEX_VERSION,
                prompt_version=Config.PROMPT_VERSION,
                prefix=Config.CACHE_KEY_PREFIX,
            )
            cached_res = redis_cache.get_cache(cache_key)
            if not cached_res and stateless_cache_key != cache_key:
                cached_res = redis_cache.get_cache(stateless_cache_key)
            if cached_res:
                rewrite = cached_res.get("rewrite_query")
                cached_sources = cached_res.get("sources", [])
                sources_from_cache = [{**item, "from_cache": True} for item in cached_sources]
                if rewrite:
                    yield f"data:{json.dumps({'rewrite_query': rewrite}, ensure_ascii=False)}\n\n"
                if sources_from_cache:
                    yield f"data:{json.dumps({'sources': sources_from_cache, 'retrieval_cache_hit': True}, ensure_ascii=False)}\n\n"
                #输出完整答案（一次性）
                answer = cached_res.get("answer", "")
                if answer:
                    yield f"data:{json.dumps({'answer_chunk': answer}, ensure_ascii=False)}\n\n"
               #发送“命中缓存”标志
                yield f"data:{json.dumps({'cache_hit': True}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n" #发送结束标志
                return
        #缓存未命中：执行 LangGraph 并流式输出
        final_answer = ""
        final_rewrite = None
        final_sources = []
        try:
            event_stream = agent_app.astream_events(inputs, version="v1")
            async for event in event_stream:
                kind = event["event"]
                name = event.get("name", "")
                node = event.get("metadata", {}).get("langgraph_node", "")
                output = event.get("data", {}).get("output", {})

                if kind == "on_chain_end" and (name in {"rewrite_node", "rewrite"} or node == "rewrite"):
                    if isinstance(output, dict) and output.get("rewrite_query"):
                        final_rewrite = output["rewrite_query"]
                        yield f"data:{json.dumps({'rewrite_query': final_rewrite}, ensure_ascii=False)}\n\n"

                if kind == "on_chain_end" and (name in {"retrieve_node", "retrieve"} or node == "retrieve"):
                    if isinstance(output, dict):
                        if output.get("retrieval_sources"):
                            final_sources = output["retrieval_sources"]
                        else:
                            final_sources = serialize_documents(output.get("documents", []))
                        retrieval_cache_hit = bool(output.get("retrieval_cache_hit", False))
                        if final_sources:
                            final_sources = [
                                {**item, "from_cache": retrieval_cache_hit}
                                for item in final_sources
                            ]
                            yield f"data:{json.dumps({'sources': final_sources, 'retrieval_cache_hit': retrieval_cache_hit}, ensure_ascii=False)}\n\n"

                if (
                    kind == "on_chat_model_stream"
                    and event["metadata"].get("langgraph_node") == "generate"
                ):
                    content = event["data"]["chunk"].content
                    if content:
                        final_answer += content
                        yield f"data:{json.dumps({'answer_chunk': content}, ensure_ascii=False)}\n\n"

                if kind == "on_chain_end" and (name in {"generate_node", "generate"} or node == "generate"):
                    if redis_cache and cache_key and final_answer:
                        cache_payload = {
                            "answer": final_answer,
                            "rewrite_query": final_rewrite,
                            "sources": final_sources,
                        }
                        redis_cache.set_json(cache_key, cache_payload, expire=Config.ANSWER_CACHE_TTL)
                        if stateless_cache_key and stateless_cache_key != cache_key:
                            redis_cache.set_json(stateless_cache_key, cache_payload, expire=Config.ANSWER_CACHE_TTL)
                    yield "data: [DONE]\n\n"

        except Exception as exc:
            error_text = str(exc)
            if "Incorrect API key" in error_text or "invalid_api_key" in error_text or "401" in error_text:
                error_text = "\u6a21\u578b\u670d\u52a1\u8ba4\u8bc1\u5931\u8d25\uff1aOPENAI_API_KEY \u65e0\u6548\u3001\u8fc7\u671f\uff0c\u6216\u4e0d\u662f\u5f53\u524d OPENAI_BASE_URL \u5bf9\u5e94\u5e73\u53f0\u7684 Key\u3002\u8bf7\u66f4\u65b0 .env \u540e\u91cd\u542f FastAPI\u3002"
            yield f"data:{json.dumps({'error': error_text}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
