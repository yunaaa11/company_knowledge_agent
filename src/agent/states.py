from typing import List, TypedDict, Annotated
import operator

class AgentState(TypedDict):
      query:str
      rewrite_query:str
      chat_history: List[dict]
      documents:List[str]
      answer:str
      # 反思结果：是否需要重新检索 (True/False)
      needs_retry:bool
      # 迭代次数，防止死循环
      loop_step: int