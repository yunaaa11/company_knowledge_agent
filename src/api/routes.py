from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.reranker import RerankProcessor
from src.retrieval.hybrid_search import HybridSearcher
from src.agent.workflow import create_graph
from langchain_openai import ChatOpenAI
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
    try:
        inputs={
            "query":request.query,
            "chat_history":request.chat_history,
            "loop_step":0
        }
        result=await agent_app.ainvoke(inputs)
        # 提取来源信息并去重
        sources=list(set([
            doc.metadata.get("source","未知来源")
            for doc in result.get("documents",[])
        ]))
        return ChatResponse(
            answer=result.get("answer","抱歉，我未能生成回答。"),
            rewrite_query=result.get("rewrite_query", ""),
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
