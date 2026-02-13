from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Document, Prefetch, FusionQuery, Filter, FieldCondition, MatchAny
import openai
from langsmith import traceable, get_current_run_tree
import json
from typing import List
from server.agents.reranker import get_reranker
from server.core.config import config

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

@traceable(name="rerank_retrieved_context", 
           description="Rerank the retrieved context using the Cohere reranker", 
           run_type="embedding")
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

def _retrieve_products_context(query: str, k: int = 5) -> dict:
    """Retrieve and rerank product context for a query."""
    qd_client = QdrantClient(url=config.qdrant_url)

    collection_name = "amazon_items-collection-hybrid-02"
    query_embeddings = create_embeddings(query)

    response = qd_client.query_points(
        collection_name=collection_name,
        prefetch=[Prefetch(
            query=query_embeddings,
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

    retrieved_context_data = {
        "context_ids": retrieved_context_ids,
        "context": retrieved_context,
        "scores": retrieved_scores,
        "context_ratings": retrieved_context_ratings
    }

    return rerank_retrieved_context(query, retrieved_context_data)


def retrieve_products(query: str) -> List[str]:
    """
    Retrieves a list of relevant product context strings from a Qdrant database using hybrid search (embedding and BM25 fusion) based on the given user query.

    Args:
        query (str): The user's search query for desired product(s).

    Returns:
        List[str]: Each string contains the product ID, description, and average rating, formatted as:
            'Product ID: <ASIN> - Description: <description> - Rating: <rating>'
    """
    
    reranked_context = _retrieve_products_context(query=query, k=5)

    reranked_retrieved_contextdata = []
    for item, context, rating in zip(reranked_context["context_ids"], 
                                     reranked_context["context"], reranked_context["context_ratings"]):
        product_context = f"Product ID: {item} - Description: {context} - Rating: {rating}"
        reranked_retrieved_contextdata.append(product_context)

    return reranked_retrieved_contextdata


def retrieve_embedding(query: str) -> List[str]:
    """Backward-compatible alias for retrieve_products."""
    return retrieve_products(query)

@traceable(name="retrieve_reviews",
           description="Retrieve reviews for a given query and product ids",
           run_type="retriever")
def retrieve_reviews(query, product_ids, k=5):
    # INSERT_YOUR_CODE
    """
    Retrieve reviews for a given query and list of product IDs.

    Args:
        query (str): The query string to search relevant reviews.
        product_ids (List[str]): A list of product IDs (parent_asin) for which reviews are to be retrieved.
        k (int, optional): The number of top reviews to retrieve. Defaults to 5.

    Returns:
        dict: A dictionary containing:
            - 'retrieved_context_ids': List of product IDs corresponding to each retrieved review.
            - 'retrieved_context': List of review texts retrieved for the query and product IDs.
            - 'similarity_scores': List of similarity scores for each retrieved review.
    """
    qdrant_client = QdrantClient(url=config.qdrant_url)
    
    collection_name = "amazon-item-collection-hybrid-01-reviews"
    k=5
    
    querry_embeddings = create_embeddings(query)
    
    try:
    
        response = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[Prefetch(
                query=querry_embeddings,
                filter=Filter(
                    must=[
                        FieldCondition(key="parent_asin", 
                                        match=MatchAny(any=product_ids))
                        
                        ]
                        
                    ),
                    limit=20
                )
            ],
            query=FusionQuery(fusion="rrf"),
            limit=k
        )
        
        retrieved_context_ids = []
        retrieved_context = []
        similarity_scores = []

        for result in response.points:
            retrieved_context_ids.append(result.payload["parent_asin"])
            retrieved_context.append(result.payload["text"])
            similarity_scores.append(result.score)
    except Exception as e:
        print(f"Error retrieving reviews: {e}")
        raise e
    
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }

@traceable(
    name="format_retrieved_reviews_context",
    run_type="prompt"
)
def process_reviews_context(context):

    formatted_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id}, review: {chunk}\n"

    return formatted_context

def get_formatted_reviews_context(query: str, item_list: list, top_k: int = 15) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.
    
    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """

    context = retrieve_reviews(query, item_list, top_k)
    formatted_context = process_reviews_context(context)

    return formatted_context
