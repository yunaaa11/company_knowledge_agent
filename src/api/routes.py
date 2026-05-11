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


class ChatRequest(BaseModel):
    query: str = Field(..., example="北京的报销标准是多少？")
    chat_history: Optional[List[dict]] = Field(default_factory=list)


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def stream_generator():
        inputs = {
            "query": request.query,
            "chat_history": request.chat_history,
            "loop_step": 0,
        }
        # 缓存键生成与命中处理
        cache_key = None
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
            cached_res = redis_cache.get_cache(cache_key)
            if cached_res:
                rewrite = cached_res.get("rewrite_query")
                if rewrite:
                    yield f"data:{json.dumps({'rewrite_query': rewrite}, ensure_ascii=False)}\n\n"
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
        async for event in agent_app.astream_events(inputs, version="v1"):
            kind = event["event"]
            
            #查询改写完成
            if kind == "on_chain_end" and event["name"] == "rewrite_node":
                final_rewrite = event["data"]["output"]["rewrite_query"]
                yield f"data:{json.dumps({'rewrite_query': final_rewrite}, ensure_ascii=False)}\n\n"
            #生成节点的 token 流
            if (
                kind == "on_chat_model_stream"
                and event["metadata"].get("langgraph_node") == "generate"
            ):
                content = event["data"]["chunk"].content
                if content:
                    final_answer += content
                    yield f"data:{json.dumps({'answer_chunk': content}, ensure_ascii=False)}\n\n"
            
            #生成节点结束 + 写入缓存
            if kind == "on_chain_end" and event["name"] == "generate_node":
                #缓存功能已开启；成功生成了有效的缓存键；得到了非空的答案
                if redis_cache and cache_key and final_answer:
                    redis_cache.set_cache(
                        cache_key,
                        {
                            "answer": final_answer,
                            "rewrite_query": final_rewrite,
                        },
                    )
                yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
