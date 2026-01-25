from qdrant_client import QdrantClient
import openai
from server.core.config import config
from langsmith import traceable, get_current_run_tree
from server.agents.models import RAGResponse
import instructor
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import Document, Prefetch, FusionQuery
from server.agents.utils.prompt_management import get_prompt_from_config
from server.agents.reranker import get_reranker

@traceable(
    name="generate_embeddings",
    description="Generate embeddings for a given query or text using OpenAI's text-embedding-3-small model",   
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model": "text-embedding-3-small"}
)
def create_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    
    current_run = get_current_run_tree()
    
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "total_tokens": response.usage.total_tokens,
            "input_tokens": response.usage.prompt_tokens,
        }
        
    return response.data[0].embedding

@traceable(name="retrieve_embedding_data", 
description="Retrieve embedding data from Qdrant for a given query and collection name",
run_type="retriever"
)
def retrieve_embedding_data(qd_client: QdrantClient, query, collection_name, k=10):
    
    querry_embeddings = create_embeddings(query)
    
    response = qd_client.query_points(
        collection_name=collection_name,
        prefetch=[Prefetch(
            query=querry_embeddings,
            using="text-embedding-3-small",
            limit=20),
            Prefetch(
                query=Document(text=query, model="qdrant/bm25"),
                using="bm25",
                limit=20)
            ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    retrieved_scores = []
    retrieved_context_ratings = []
    
    for point in response.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["description"])
        retrieved_scores.append(point.score)
        retrieved_context_ratings.append(point.payload["average_rating"])

    # return dictionary of retrieved data
    return {
        "context_ids": retrieved_context_ids,
        "context": retrieved_context,
        "scores": retrieved_scores,
        "context_ratings": retrieved_context_ratings
    }

@traceable(name="rerank_retrieved_context", 
           description="Rerank the retrieved context using the Cohere reranker", 
           run_type="reranker")
def rerank_retrieved_context(query,retrieved_context):
    reranker = get_reranker(provider="cohere")
    context_list = retrieved_context["context"]
    reranked_context = reranker.rerank(query=query, documents=context_list, top_n=5)
    
    reranked_retrieved_context_ids = []
    reranked_retrieved_context = []
    reranked_retrieved_scores = []
    reranked_retrieved_context_ratings = []
    
    for new_context in reranked_context:
        index = new_context.get("index")
        reranked_retrieved_context_ids.append(retrieved_context["context_ids"][index])
        reranked_retrieved_context.append(retrieved_context["context"][index])
        reranked_retrieved_scores.append(retrieved_context["scores"][index])
        reranked_retrieved_context_ratings.append(retrieved_context["context_ratings"][index])
    
    #return only top 5 results    
    return {
        "context_ids": reranked_retrieved_context_ids,
        "context": reranked_retrieved_context,
        "scores": reranked_retrieved_scores,
        "context_ratings": reranked_retrieved_context_ratings
    }

@traceable(
    name="format_context",
    description="Format the retrieved context into a string",
    run_type="retriever"
)
def format_context(retrived_context):
    formatted_context = ""
    for id, chunk, rating in zip(retrived_context["context_ids"], retrived_context["context"], retrived_context["context_ratings"]):
        formatted_context += f"Product ID: {id}, rating: {rating}, description: {chunk.strip()}\n"
    return formatted_context

@traceable(name="construct_prompt", run_type="prompt")
def build_prompt(preprocessed_context, question):
    template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/rag_system.yml', 'retrieval_generation')
    prompt = template.render(preprocessed_context=preprocessed_context, question=question)
    return prompt

@traceable(name="generate_llm_response",
description="Generate a response from the LLM using the prompt",
run_type="llm",
metadata={"ls_provider": "openai", "ls_model_name": "gpt-5-nano"}
)
def generate_llm_response(prompt, model="gpt-4.1-mini"):
    
    client = instructor.from_openai(openai.OpenAI())
    
    response, raw_response = client.chat.completions.create_with_completion(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
        ],
        response_model=RAGResponse
    )
    
    current_run = get_current_run_tree()
    
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
            "model": model
        }
        
    return response

@traceable(
    name="integrated_rag_pipeline",
    description="Integrate the RAG pipeline for a given question",
)
def integrated_rag_pipeline(question, model="gpt-4.1-mini", top_k=10):
    
    qdrant_client = QdrantClient(   
        url=config.qdrant_url,
    )
    # Step 1: Retrieve relevant context
    retrieved_context = retrieve_embedding_data(
        qdrant_client,
        question,
        collection_name="amazon_items-collection-hybrid-02",
        k=top_k
    )
    # step 1.1: Rerank the retrieved context
    reranked_context = rerank_retrieved_context(question, retrieved_context)
    
    # Step 2: Format context
    formatted_context = format_context(reranked_context)   
    # Step 3: Build prompt
    prompt = build_prompt(formatted_context, question)
    # Step 4: Generate response
    response = generate_llm_response(prompt, model)
    
    final_response = {
        "question": question,
        "answer": response.answer,
        "references": response.references,
        "retrieved_context_ids": retrieved_context["context_ids"],
        "retrieved_context": retrieved_context["context"],
        "similarity_scores": retrieved_context["scores"],
    }
        
    return final_response

def rag_pipeline_wrapper(question, top_k=10):
    
    qdrant_client = QdrantClient(   
        url=config.qdrant_url,
    )
    
    result = integrated_rag_pipeline(question, model="gpt-4.1-mini", top_k=top_k)
    
    used_context = []
    
    dummy_vector = np.zeros(1536).tolist()
        
    for item in result.get("references"):
        payload = qdrant_client.query_points(
            collection_name="amazon_items-collection-hybrid-02",
            query=dummy_vector,
            limit=1,
            with_payload=True,
            using="text-embedding-3-small",
            with_vectors=False,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        )
        if payload.points[0].payload["parent_asin"]:
            image_url = payload.points[0].payload.get("image", None)
            price = payload.points[0].payload.get("price", None)
            if image_url:
                used_context.append({
                    "id": item.id,
                    "description": item.description,
                    "image_url": image_url,
                    "price": price
                })
            
    return {
        "answer": result.get("answer", ""),
        "used_context": used_context,
    }
    
    