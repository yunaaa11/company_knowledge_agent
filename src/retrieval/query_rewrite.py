from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import Config
class QueryRewriter:
    # 查询重写→ 独立、完整的搜索词
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            "根据以下对话历史和当前问题，重新编写一个适合检索的独立搜索词：\n"
            "历史：{chat_history}\n"
            "问题：{query}"
        )

    def rewrite(self, query: str, chat_history: str = "") -> str:
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query, "chat_history": chat_history})
        return response.content