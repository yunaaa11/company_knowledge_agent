import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.reranker import RerankProcessor
from src.retrieval.hybrid_search import HybridSearcher
from src.agent.workflow import create_graph
from langchain_openai import ChatOpenAI
from fastapi.responses import StreamingResponse
from config import Config

router=APIRouter()
#全局单例初始化
vm = VectorStoreManager()
hs = HybridSearcher(vm)
reranker = RerankProcessor(hs.get_ensemble_retriever())
llm = ChatOpenAI(
    model=Config.LLM_MODEL,
    openai_api_key=Config.OPENAI_API_KEY,
    openai_api_base=Config.OPENAI_BASE_URL,
    temperature=0
)
agent_app = create_graph(vm, reranker, llm)
#数据模型定义
class ChatRequest(BaseModel):
    query: str = Field(..., example="北京的报销标准是多少？")
    chat_history: Optional[List[dict]] = Field(default_factory=list)
class ChatResponse(BaseModel):
    answer:str
    rewrite_query:str
    sources:List[str]
#路由接口
@router.post("/chat",response_model=ChatResponse)
async def chat_endpoint(request:ChatRequest):
     # 1. 定义内部异步生成器
    async def stream_generator():
        inputs={
            "query":request.query,
            "chat_history":request.chat_history,
            "loop_step":0
        }
        # 使用 astream_events 捕获 LLM 实时生成的 token
        async for event in agent_app.astream_events(inputs,version="v1"):
            kind=event["event"]
            # 捕获改写后的问题（用于前端展示）
            if kind=="on_chain_end" and event["name"]=="rewrite_node":
                rewrite=event["data"]["output"]["rewrite_query"]
                yield f"data:{json.dumps({'rewrite_query':rewrite})}\n\n"
            # 捕获生成的 Token (核心流式输出)
            if kind=="on_chat_model_stream" and event["metadata"].get("langgraph_node")=="generate":
                content=event["data"]["chunk"].content
                if content:
                    yield f"data:{json.dumps({'answer_chunk':content}, ensure_ascii=False)}\n\n"
            #捕获最终引用的来源
            if kind=="on_chain_end" and event["name"]=="generate_node":
                # 这里可以从 state 提取 documents 传给前端
                yield f"data: [DONE]\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
