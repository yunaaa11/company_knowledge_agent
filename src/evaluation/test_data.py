from langchain_community.document_loaders import DirectoryLoader
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
     MultiHopSpecificQuerySynthesizer,
)

class TestDataGenerator:
    def __init__(self, llm, embeddings):
        # 内部会初始化一个可生成测试集的引擎
        self.generator = TestsetGenerator.from_langchain(
            llm=llm,
            embedding_model=embeddings
        )
        self.llm = llm

    def generate_from_docs(self, documents, test_size=3):
        """生成测试集"""
        # 生成测试集
        testset = self.generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=test_size,
            # query_distribution=query_distribution,
            with_debugging_logs=False
        )
        return testset.to_pandas()