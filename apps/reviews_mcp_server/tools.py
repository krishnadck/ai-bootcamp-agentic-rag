from qdrant_client import QdrantClient
from qdrant_client.models import Document, Prefetch, FusionQuery, Filter, FieldCondition, MatchAny
import openai
from typing import List
from core.config import config

def create_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    
    return response.data[0].embedding


def retrieve_reviews(query, product_ids, k=5):
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