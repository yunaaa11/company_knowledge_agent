from src.retrieval.query_rewrite import QueryRewriter
from src.retrieval.reranker import RerankProcessor
from src.agent.states import AgentState
class Nodes:
    def __init__(self,vector_manager,reranker,llm):
        self.rewriter=QueryRewriter(llm=llm)
        self.reranker=reranker
        self.llm = llm

    async def rewrite_node(self,state:AgentState):
        print("--- 正在改写问题 ---")
        chat_history = state.get("chat_history")   # 从状态中获取历史
        new_query=await self.rewriter.rewrite(state["query"], chat_history=chat_history)
        return {"rewrite_query": new_query, "loop_step": state.get("loop_step", 0) + 1}
    
    async def retrieve_node(self,state:AgentState):
        print("--- 正在执行深度检索 ---")
        queries=state["rewrite_query"]
        if isinstance(queries, str):
            queries = [queries]

        all_docs = []
        # 循环检索
        for q in queries:
            print(f"  🔍 正在检索子查询: {q}")
        # 兼容处理：优先使用 retrieve 方法，否则使用 invoke
            if hasattr(self.reranker, "retrieve"):
                docs = self.reranker.retrieve(q)
            else:
            # 假设是 retriever（如 EnsembleRetriever），调用 invoke 获取文档列表
                docs = self.reranker.invoke(q)
            all_docs.extend(docs)   
        return {"documents":all_docs}
    
    async def generate_node(self,state:AgentState):
        print("--- 正在生成回答 ---")
        context="\n".join([d.page_content for d in state["documents"]])
        system_prompt = (
    "你是一个严谨的企业行政助手，负责回答员工关于公司制度的问题。\n"
    "请严格依据以下资料回答问题，不得编造或使用外部知识。\n\n"
    "【可用制度文档】\n"
    "1. 《员工请假管理制度》（人力资源部）\n"
    "2. 《员工报销管理制度》（财务部）\n"
    "3. 《IT 故障处理指南》（IT部）\n"
    "4. 《办公用品申领流程》（行政部）\n"
    "5. 《信息安全行为规范》（IT部/合规部）\n"
    "6. 《员工绩效管理制度》（人力资源部）\n\n"
    "【冲突处理规则】\n"
    "如果不同部门/文档对同一事项的规定存在冲突，请按以下优先级采纳：\n"
    "- 最高优先级：公司层面的强制性规范（如信息安全行为规范）\n"
    "- 次优先级：人力资源部发布的制度（请假、绩效）\n"
    "- 第三优先级：财务部（报销）、行政部（办公用品）\n"
    "- 最低优先级：IT部操作指南（仅作参考，不与其他部门强制性规则冲突）\n"
    "若无法判断优先级，请如实列出不同规定，并提示用户以最新发布的正式制度为准。\n\n"
    "【回答要求】\n"
    "1. 引用具体条款时，注明来源文档名称（例如：根据《员工请假管理制度》第四条）。\n"
    "2. 如果检索到的资料不足以回答问题，请明确说“资料中未找到相关信息”。\n"
    "3. 回答应简洁、结构化，可使用分点或表格帮助理解。\n"
    "4. 对于涉及金额、天数、百分比等具体数字，务必核对准确。\n"
    "5. 禁止给出超出制度范围的建议（如“可以申请更多年假”）。\n"
)
        prompt = f"{system_prompt}\n\n根据资料：{context} 回答：{state['rewrite_query']}"
        response = await self.llm.ainvoke(prompt) 
        return {"answer": response.content}
