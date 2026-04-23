from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import pandas as pd
class RagasEvaluator:
    def __init__(self,llm):
        self.llm=llm
        self.metrics=[
            faithfulness,        # 忠实度：回答是否源自文档
            answer_relevancy,    # 相关性：回答是否对题
            context_precision,   # 上下文精度：检索出的文档是否精准
            context_recall       # 上下文召回：是否包含标准答案的关键信息
        ]
    def evaluate_response(self,query:str,state_output:dict,ground_truth:str):
        """
        对单次 Agent 输出进行评估
        state_output: 节点的输出字典，包含 documents 和 answer
        """
        data={
            "question":[query],
            "answer":[state_output.get("answer","")],
            # 将 LangChain 的 Document 对象转为纯文本列表
            "contexts":[[doc.page_content for doc in state_output.get("documents",[])]],
            #人工标注或期望的标准答案
            "ground_truth":[ground_truth]
        }
        dataset=Dataset.from_dict(data)
        #执行评估
        result=evaluate(dataset,metrics=self.metrics, llm=self.llm)
        df=result.to_pandas()
     # 确保包含 question 和 ground_truth 列（某些版本可能不自动保留）
        if 'question' not in df.columns:
            df.insert(0, 'question', query)
        if 'ground_truth' not in df.columns:
            df.insert(1, 'ground_truth', ground_truth)
        return df
