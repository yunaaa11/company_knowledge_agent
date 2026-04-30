from langgraph.graph import StateGraph, END
from src.agent.states import AgentState
from src.agent.agent_nodes import Nodes
from src.agent.reflection import Reflection
def create_graph(vector_manager, reranker, llm, enable_rewrite=True, enable_reflection=True):
    workflow=StateGraph(AgentState)
    nodes=Nodes(vector_manager,reranker,llm)
    # 1. 添加节点
    workflow.add_node("retrieve",nodes.retrieve_node)
    workflow.add_node("generate",nodes.generate_node)
    if enable_rewrite:
        workflow.add_node("rewrite",nodes.rewrite_node)
    # 2. 设置连线
    #可选节点：改写。如果关闭，则直接从检索开始
    if enable_rewrite:
        workflow.set_entry_point("rewrite")
        workflow.add_edge("rewrite", "retrieve")
    else:
        workflow.set_entry_point("retrieve")
    # 3. 添加条件边（反思环节）
    if enable_reflection and enable_rewrite:
        workflow.add_conditional_edges(
            "retrieve",
            Reflection.grade_documents,
            {
                "retry":"rewrite", # 质量不好：跳回改写
                "generate":"generate" # 质量好：去生成答案
            }
        )
    else:
        workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate",END)
    return workflow.compile()


