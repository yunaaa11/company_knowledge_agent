from typing import List, TypedDict, Union

class AgentState(TypedDict):
      query:str
      rewrite_query:Union[str, List[str]]
      chat_history: List[dict]
      documents:list
      answer:str
      # 反思结果：是否需要重新检索 (True/False)
      needs_retry:bool
      # 迭代次数，防止死循环
      loop_step: int