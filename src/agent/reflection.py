from src.agent.states import AgentState
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Grade(BaseModel):
    binary_score:str=Field(description="文档是否相关，'yes' or 'no'")

class Reflection:
    @staticmethod
    def grade_documents(state:AgentState):
        """
        简单规则：基于重排分数检查前2条文档是否 ≥0.2
        """
        print("--- 正在评估检索质量 ---")
        docs = state.get("documents", [])
        loop_step = state.get("loop_step", 0)
        
         # 规则1：没有文档且重试次数 <= 2 → 重试
        if not docs and loop_step <= 2:
            return "retry"

        top_docs = docs[:2]
        high_score_docs = [
            doc for doc in top_docs
            if doc.metadata.get("relevance_score", 0.0) >= 0.2
        ]
        # 规则2：前2条中没有任何一条得分 >= 0.2，且重试次数 <= 2 → 重试
        if not high_score_docs and loop_step <= 2:
            print("--- Top 文档相关性偏低，触发重试 ---")
            return "retry"
        #其他（有高相关文档，或重试已达上限）
        return "generate"
    @staticmethod
    async def grade_documents_complex(state:AgentState,llm):
        """
        高级版：用 LLM 判断前5条文档的相关率 ≥0.4
        
        参数:
            state: AgentState 当前状态
            llm: ChatOpenAI 实例，要求支持结构化输出
            
        返回:
            "retry" 或 "generate"
        """
        print("--- 正在深度评估检索质量 ---")
        docs = state.get("documents", [])
        if not docs:
            # 无文档且 loop_step <= 2 → retry，否则 generate
            loop_step = state.get("loop_step", 0)
            if loop_step <= 2:
                print("--- 无文档且重试次数未达上限，触发重试 ---")
                return "retry"
            else:
                print("--- 无文档但已达重试上限，强行进入生成 ---")
                return "generate"
            
        query = state.get("rewrite_query", state.get("query", ""))
        loop_step = state.get("loop_step", 0)

        # 构造一个结构化输出链，返回 Grade (binary_score='yes'/'no')
        prompt=ChatPromptTemplate.from_template(
            "你是一个质检员。判断以下文档是否能回答用户问题：\n"
            "用户问题: {query}\n"
            "文档片段: {context}\n"
            "请只回答 'yes' 或 'no'，表示文档是否能回答问题。"    
        )
        scorer=prompt|llm.with_structured_output(Grade)

        relevant_count=0
        # 为了性能，可以只抽取前 3-5 条重排后的文档进行抽检
        check_docs=docs[:5]

        for doc in check_docs:
            res=await scorer.ainvoke({"query":query,"context":doc.page_content})
            if res.binary_score=="yes":
                relevant_count+=1
        # 计算相关率
        relevance_rate = relevant_count / len(check_docs) if check_docs else 0
        if relevance_rate < 0.4 and state["loop_step"] <= 2:
            print(f"--- 质量过低 ({relevance_rate})，触发重试 ---")
            return "retry"
        else:
            print("--- 相关率达标或已达重试上限，进入生成 ---")
            return "generate"


