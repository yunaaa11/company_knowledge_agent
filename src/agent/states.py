from typing import List, Optional, TypedDict, Union

class AgentState(TypedDict):
      query:str
      rewrite_query:Union[str, List[str]]
      hyde_query:str
      chat_history: List[dict]
      documents:list
      retrieval_sources:list
      retrieval_cache_hit: bool
      retrieval_query_stats: dict
      answer:str
      # 反思结果：是否需要重新检索 (True/False)
      needs_retry:bool
      # 迭代次数，防止死循环
      loop_step: int
      # 上一轮检测到的意图（用于追问继承）
      prev_intent: Optional[str]
      # 上一轮检索的高分文档（用于追问范围约束）
      prev_documents: list
