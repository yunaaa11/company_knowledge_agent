from langgraph.graph import StateGraph, END
from src.agent.states import AgentState
from src.agent.agent_nodes import Nodes
from src.agent.reflection import Reflection
def create_graph(vector_manager,reranker,llm):
    workflow=StateGraph(AgentState)
    nodes=Nodes(vector_manager,reranker,llm)
    # 1. 添加节点
    workflow.add_node("rewrite",nodes.rewrite_node)
    workflow.add_node("retrieve",nodes.retrieve_node)
    workflow.add_node("generate",nodes.generate_node)
    # 2. 设置连线
    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")
    # 3. 添加条件边（反思环节）
    workflow.add_conditional_edges(
        "retrieve",
        Reflection.grade_documents,
        {
            "retry":"rewrite", # 质量不好：跳回改写
            "generate":"generate" # 质量好：去生成答案
        }
    )
    workflow.add_edge("generate",END)
    return workflow.compile()


