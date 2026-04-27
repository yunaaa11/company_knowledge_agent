from src.agent.states import AgentState
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
class Grade(BaseModel):
    binary_score:str=Field(description="文档是否相关，'yes' or 'no'")

class Reflection:
    @staticmethod
    def grade_documents(state:AgentState):
        """
        这个函数作为条件边
        决定是去 generate还是去 rewrite
        """
        print("--- 正在评估检索质量 ---")
        docs = state["documents"]
        
        # 实际项目中这里通常再调一次轻量级 LLM 进行打分
        # 简化逻辑：如果没有搜到文档，或者迭代次数 < 2，则重试
        if not docs and state["loop_step"] <= 2:
            return "retry"
        else:
            return "generate"
    @staticmethod
    async def grade_documents_complex(state:AgentState,llm):
        """
        复杂版质量评估：使用 LLM 对检索到的文档片段进行相关性打分，
        根据相关率决定是否重试，同时考虑 loop_step 限制。
        
        参数:
            state: AgentState 当前状态
            llm: ChatOpenAI 实例，要求支持结构化输出
            
        返回:
            "retry" 或 "generate"
        """
        print("--- 正在深度评估检索质量 ---")
        docs = state.get("documents", [])
        if not docs:
            # 没有文档，直接判断是否可以重试
            loop_step = state.get("loop_step", 0)
            if loop_step <= 2:
                print("--- 无文档且重试次数未达上限，触发重试 ---")
                return "retry"
            else:
                print("--- 无文档但已达重试上限，强行进入生成 ---")
                return "generate"
            
        query = state.get("rewrite_query", state.get("query", ""))
        loop_step = state.get("loop_step", 0)

        # 构造打分链
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


