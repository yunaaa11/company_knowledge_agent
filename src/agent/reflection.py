from src.agent.states import AgentState
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

