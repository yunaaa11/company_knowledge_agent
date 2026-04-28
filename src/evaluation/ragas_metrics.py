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
        #提取基础数据
        documents = state_output.get("documents", [])
        answer = state_output.get("answer", "")
        contexts = [doc.page_content for doc in documents] # 纯文本上下文列表
        sources = [doc.metadata.get("source", "未知") for doc in documents]  # 来源文档名
        scores = [float(doc.metadata.get("relevance_score", 0.0)) for doc in documents]  # 重排后分数
        
        #构建Ragas数据集格式
        data={
            "question":[query],
            "answer":[answer],
            # 将 LangChain 的 Document 对象转为纯文本列表
            "contexts":[contexts],
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
        
        #计算自定义指标
        #所有文档重排分的均值
        avg_relevance = round(sum(scores) / len(scores), 4) if scores else 0.0
        top_relevance = round(scores[0], 4) if scores else 0.0
        unique_sources = len(set(sources)) #来源文档的去重数量
        keyword_hits = self._keyword_hit_ratio(ground_truth, answer)
        strict_score = self._strict_score(df.iloc[0], top_relevance, avg_relevance, keyword_hits)

        df["retrieval_count"] = len(documents)
        df["unique_source_count"] = unique_sources
        df["retrieved_sources"] = " | ".join(sources)
        df["top_relevance_score"] = top_relevance
        df["avg_relevance_score"] = avg_relevance
        df["keyword_hit_ratio"] = keyword_hits
        df["strict_score"] = strict_score
        df["answer"] = answer
        return df

    @staticmethod
    def _keyword_hit_ratio(ground_truth: str, answer: str) -> float:
        """计算答案中包含标准答案中关键词的比例"""
        tokens = []
        for chunk in ground_truth.replace("，", " ").replace("。", " ").replace("；", " ").split():
            token = chunk.strip(" 1234567890.、:：()（）")
            if len(token) >= 2:
                tokens.append(token)
        if not tokens:
            return 0.0

        hits = sum(1 for token in set(tokens) if token in answer)
        return round(hits / len(set(tokens)), 4)

    @staticmethod
    def _strict_score(row: pd.Series, top_relevance: float, avg_relevance: float, keyword_hit_ratio: float) -> float:
        """计算一个综合严格分数，融合Ragas的四个指标和检索质量指标"""
        weighted_metrics = [
            ("faithfulness", 0.35),
            ("answer_relevancy", 0.25),
            ("context_precision", 0.2),
            ("context_recall", 0.2),
        ]
        metric_sum = 0.0
        weight_sum = 0.0
        for metric_name, weight in weighted_metrics:
            value = row.get(metric_name, 0.0)
            if pd.isna(value):
                continue
            metric_sum += float(value) * weight
            weight_sum += weight
        # 当某些RAGAS指标缺失时，按有效权重做归一化，避免 strict_score 变成 NaN
        ragas_score = metric_sum / weight_sum if weight_sum > 0 else 0.0
        retrieval_score = top_relevance * 0.5 + avg_relevance * 0.3 + keyword_hit_ratio * 0.2
        return round(ragas_score * 0.75 + retrieval_score * 0.25, 4)
