"""Evaluate a LangSmith RAG dataset with Ragas and log scores back to LangSmith."""

from typing import Dict, Any
from openai import AsyncOpenAI
from langsmith import Client, aevaluate
import asyncio
from ragas import SingleTurnSample
from ragas.metrics.collections import(
    AnswerRelevancy,
    Faithfulness,
)
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from src.server.agents.retrieval_generation import integrated_rag_pipeline

class RagasLangSmithEvaluator:
    
    def __init__(self):
        self.client = AsyncOpenAI()
        self.llm = llm_factory(model="gpt-4o-mini", client=self.client)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", client=self.client)
        
        self.m_faithfulness = Faithfulness(llm=self.llm)
        self.m_relevancy = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.m_precision = IDBasedContextPrecision()
        self.m_recall = IDBasedContextRecall()

    async def evaluate(self, run: any, example: any) -> Dict[str, Any]:
        user_input = example.inputs.get("question")
        
        # run the RAG pipeline
        response = integrated_rag_pipeline(user_input)
        
        response_text = run.outputs.get("answer")
        retrieved_contexts = run.outputs.get("retrieved_context")
        retrieved_context_ids = run.outputs.get("retrieved_context_ids")

        ref_truth = example.outputs.get("ground_truth")
        ref_ids = example.outputs.get("reference_chunks", [])        
        
        # create a sample for the metrics
        sample = SingleTurnSample(
            user_input=user_input,
            response = response_text,
            retrieved_context_ids=retrieved_context_ids,
            retrieved_contexts=retrieved_contexts,
            reference = ref_truth,
            reference_context_ids = ref_ids,
        )
        
        results = {}
        
        results["context_precision"] = self.m_precision.single_turn_score(sample)
        results["context_recall"] = self.m_recall.single_turn_score(sample)
        
        if retrieved_contexts:
            results["faithfulness"] = await self.m_faithfulness.ascore(
                user_input=user_input,
                response=response_text,
                retrieved_contexts=retrieved_contexts,
                )
        else:
            results["faithfulness"] = 0.0

        results["answer_relevancy"] = await self.m_relevancy.ascore(user_input=user_input,
                response=response_text,
                )
        
        return {
            "results": [
                {"key": "ragas_faithfulness", "score": results["faithfulness"]},
                {"key": "ragas_answer_relevancy", "score": results["answer_relevancy"]},
                {"key": "ragas_context_precision", "score": results["context_precision"]},
                {"key": "ragas_context_recall", "score": results["context_recall"]},
            ]
        }

import nest_asyncio
nest_asyncio.apply()
     
async def query_rag_system(inputs: dict) -> dict:
    
    question = inputs["question"]
    
    response = integrated_rag_pipeline(question)
    
    return response

async def main():
    dataset_name = "rag-evaluation-dataset"
    
    #to test for few samples
    #client = Client()
    #all_examples = list(client.list_examples(dataset_name=dataset_name))    
    #test_examples = all_examples[:2]
    
    ragas_evaluator = RagasLangSmithEvaluator()
    
    print("Evaluating dataset: ", dataset_name)
    
    results = await aevaluate(
        query_rag_system,
        data=dataset_name,
        evaluators=[ragas_evaluator.evaluate],
        experiment_prefix="Ragas-rag-pipeline-evaluation-01",
        max_concurrency=10
    )

    print("Results: ", results)

if __name__ == "__main__":
    asyncio.run(main())