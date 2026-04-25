import sys
import os

# 1. 环境路径设置
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.reranker import RerankProcessor
from src.agent.workflow import create_graph 
from src.retrieval.hybrid_search import HybridSearcher
from langchain_openai import ChatOpenAI
from src.cache.redis_client import RedisCache
from config import Config

def run_interactive_session():
    #初始化所有组件
    print("--- 正在初始化系统（加载向量库与混合检索） ---")
    llm=ChatOpenAI(model=Config.LLM_MODEL,temperature=0)
    vm=VectorStoreManager()
    hs=HybridSearcher(vm)
    base_retriever=hs.get_ensemble_retriever()
    reranker=RerankProcessor(base_retriever)
    app=create_graph(vm,reranker,llm)  
    # app = create_graph(vm, base_retriever, llm)
    
    redis_cache=RedisCache()
    chat_history = []
    print("\n✅ 系统就绪！你可以开始提问了（输入 'exit' 退出）")
    print("-" * 50)

    while True:
        query = input("\n提问: ").strip()
        if query.lower() in ['exit', 'quit', '退出']:
            break
        if not query:
            continue
        # --- 缓存检查逻辑 ---
        cache_hit=False
        if Config.ENABLE_CACHE:
            # 使用 query 生成唯一的缓存键
            cache_key = redis_cache.generate_query_key(query)
            cached_res = redis_cache.get_cache(cache_key)
            #这里缓存的 last_node_data 是整个节点输出字典，但取回答时只用了 answer
            if cached_res:
                print("⚡ [Redis] 命中缓存，直接返回结果...")
                answer=cached_res.get("answer")
                print(f"🤖 行政助手: {answer}")
                # 更新历史以便后续对话
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})
                cache_hit = True
        #避免执行耗时的检索+LLM
        if cache_hit:
            continue
        
        #Agent流程
        inputs={"query":query,"chat_history":chat_history,"loop_step":0}
        final_answer=""
        last_node_data={}
        #流式输出节点过程
        for output in app.stream(inputs):
            for key,value in output.items():
                print(f"⚙️  进入节点: [{key}]")
                last_node_data = value # 记录最后一个节点的数据
        #提取回答
        if "answer" in last_node_data:
            final_answer=last_node_data["answer"]
            print(f"\n🤖 行政助手: {final_answer}")
            #写入缓存
            if Config.ENABLE_CACHE:
                redis_cache.set_cache(cache_key,last_node_data)
                #更新对话历史
                chat_history.append({"role":"user","content":query})
                chat_history.append({"role":"assistant","content":final_answer})
            else:
                print("❌ 未能生成有效回答，请检查检索质量。")
if __name__=="__main__":
       run_interactive_session()

