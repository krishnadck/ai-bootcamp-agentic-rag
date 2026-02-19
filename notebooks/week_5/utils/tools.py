from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Document, Prefetch, FusionQuery, Filter, FieldCondition, MatchAny, MatchValue
import openai
from langsmith import traceable, get_current_run_tree
from typing import List
from server.core.config import config

from abc import ABC, abstractmethod
from typing import Dict, Any

# Optional imports to prevent crashes if libraries aren't installed
try:
    import cohere
except ImportError:
    cohere = None

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None

class BaseReranker(ABC):
    """Abstract interface that all rerankers must follow."""
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        pass

class CohereReranker(BaseReranker):
    def __init__(self, model: str = "rerank-v4.0-fast"):
        # Automatically load key from environment
        self.client = cohere.ClientV2()
        self.model = model

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n
        )
        
        # Standardize output to match our generic format
        return [
            {
                "index": result.index,
                "text": documents[result.index],
                "score": result.relevance_score,
                "provider": "cohere"
            }
            for result in response.results
        ]

class FlashRankReranker(BaseReranker):
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        if not Ranker:
            raise ImportError("FlashRank library not found. Run: pip install flashrank")
        
        # Loads model into CPU memory (takes ~1 sec once)
        self.ranker = Ranker(model_name=model_name, cache_dir="/opt")

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []

        # FlashRank requires a specific input format: [{"id": 1, "text": "..."}]
        passages = [
            {"id": i, "text": doc} 
            for i, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # Standardize output
        return [
            {
                "index": result["id"],
                "text": result["text"],
                "score": result["score"],
                "provider": "flashrank"
            }
            for result in results[:top_n]
        ]

def get_reranker(provider: str = "cohere") -> BaseReranker:
    """Factory function to get the desired reranker."""
    if provider.lower() == "cohere":
        return CohereReranker()
    elif provider.lower() == "flashrank":
        return FlashRankReranker()
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")


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


@traceable(name="add_to_shopping_cart",
           description="Add a list of provided items to the shopping cart",
           run_type="tool")
def add_to_shopping_cart(items: List[Dict[str, Any]], user_id: str, cart_id: str) -> Dict[str, Any]:
    """Add a list of provided items to the shopping cart.
        Reads the product details qdrant collection "amazon_items-collection-hybrid-02"
        and adds the items to the shopping cart.
    
    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.
        
    Returns:
        A dictionary with status message and items with updated quantities.
    """
    import psycopg2
    from psycopg2 import Error as PsycopgError

    if not items:
        return {
            "success": False,
            "message": "No items provided.",
            "items": [],
            "skipped_items": [],
        }

    qdrant_client = QdrantClient(url=config.qdrant_url)
    collection_name = "amazon_items-collection-hybrid-02"

    valid_items = []
    skipped_items = []
    for item in items:
        pid = item.get("product_id")
        qty = item.get("quantity")
        if not pid or not isinstance(qty, int) or qty <= 0:
            skipped_items.append(
                {
                    "item": item,
                    "reason": "Invalid item: product_id is required and quantity must be a positive integer.",
                }
            )
            continue
        valid_items.append({"product_id": pid, "quantity": qty})

    if not valid_items:
        return {
            "success": False,
            "message": "No valid items to add or update.",
            "items": [],
            "skipped_items": skipped_items,
        }

    product_ids = [item["product_id"] for item in valid_items]

    try:
        qdrant_results = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                should=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=pid)
                    )
                    for pid in product_ids
                ]
            ),
            limit=max(100, len(product_ids)),
        )[0]
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to fetch product details from Qdrant: {str(e)}",
            "items": [],
            "skipped_items": skipped_items,
        }

    product_detail_map = {}
    for product in qdrant_results:
        if not hasattr(product, "payload"):
            continue
        parent_asin = product.payload.get("parent_asin")
        if parent_asin:
            product_detail_map[parent_asin] = product.payload

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(
            dbname="tools_database",
            user="langgraph_user",
            password="langgraph_password",
            host="localhost",
            port=5432,
        )
        conn.autocommit = False
        cur = conn.cursor()

        upserted_product_ids = []
        for item in valid_items:
            pid = item["product_id"]
            qty = item["quantity"]
            details = product_detail_map.get(pid)

            if not details:
                skipped_items.append(
                    {
                        "item": item,
                        "reason": f"Product details not found in Qdrant for product_id '{pid}'.",
                    }
                )
                continue

            price = details.get("price")
            currency = details.get("currency", "USD")
            image_url = details.get("image", "")

            cur.execute(
                """
                INSERT INTO shopping_carts.shopping_cart_items
                (user_id, shopping_cart_id, product_id, price, quantity, currency, product_image_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, shopping_cart_id, product_id)
                DO UPDATE
                SET quantity = shopping_carts.shopping_cart_items.quantity + EXCLUDED.quantity,
                    price = EXCLUDED.price,
                    currency = EXCLUDED.currency,
                    product_image_url = EXCLUDED.product_image_url
                """,
                (user_id, cart_id, pid, price, qty, currency, image_url),
            )
            upserted_product_ids.append(pid)

        if upserted_product_ids:
            cur.execute(
                """
                SELECT product_id, quantity, price, currency, product_image_url, (price * quantity) AS total_price
                FROM shopping_carts.shopping_cart_items
                WHERE user_id=%s AND shopping_cart_id=%s AND product_id = ANY(%s)
                ORDER BY updated_at DESC, id DESC
                """,
                (user_id, cart_id, upserted_product_ids),
            )
            rows = cur.fetchall()
        else:
            rows = []

        conn.commit()

        return {
            "success": True,
            "message": f"Added/Updated {len(rows)} item(s) in shopping cart.",
            "items": [
                {
                    "product_id": row[0],
                    "quantity": row[1],
                    "price": float(row[2]) if row[2] is not None else None,
                    "currency": row[3],
                    "product_image_url": row[4],
                    "total_price": float(row[5]) if row[5] is not None else None,
                }
                for row in rows
            ],
            "skipped_items": skipped_items,
        }
    except PsycopgError as e:
        if conn is not None:
            conn.rollback()
        return {
            "success": False,
            "message": f"Database error while adding items to shopping cart: {str(e)}",
            "items": [],
            "skipped_items": skipped_items,
        }
    except Exception as e:
        if conn is not None:
            conn.rollback()
        return {
            "success": False,
            "message": f"Unexpected error while adding items to shopping cart: {str(e)}",
            "items": [],
            "skipped_items": skipped_items,
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@traceable(name="read_shopping_cart",
           description="Read the shopping cart for a given userID and cart_id",
           run_type="tool")
def read_shopping_cart(user_id: str, cart_id: str) -> list[Dict[str, Any]]:
    """Read the shopping cart for a given userID and cart_id.

    Args:
        user_id: The id of the user to read the shopping cart for.
        cart_id: The id of the shopping cart to read.
    Returns:
        A list of dictionaries with the following keys: product_id, quantity, price, currency, product_image_url.

    """
    import psycopg2

    conn = psycopg2.connect(
        dbname="tools_database",
        user="langgraph_user",
        password="langgraph_password",
        host="localhost",
        port=5432,
    )
    cur = conn.cursor()

    cur.execute(
        """
        SELECT product_id, quantity, price, currency, product_image_url, (price * quantity) as total_price
        FROM shopping_carts.shopping_cart_items
        WHERE user_id=%s AND shopping_cart_id=%s
        ORDER BY updated_at DESC, id DESC
        """,
        (user_id, cart_id),
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "product_id": row[0],
            "quantity": row[1],
            "price": float(row[2]) if row[2] is not None else None,
            "currency": row[3],
            "product_image_url": row[4],
            "total_price": float(row[5]) if row[5] is not None else None,
        }
        for row in rows
    ]


@traceable(name="remove_item_from_cart",
           description="Remove one product from a user's shopping cart",
           run_type="tool")
def remove_item_from_cart(product_id: str, user_id: str, cart_id: str) -> bool:
    """Remove one product from a user's shopping cart.

    Args:
        product_id: Product to remove from cart.
        user_id: User id who owns the cart.
        cart_id: Shopping cart id.

    Returns:
        True if a row was deleted, else False.
    """
    import psycopg2

    conn = psycopg2.connect(
        dbname="tools_database",
        user="langgraph_user",
        password="langgraph_password",
        host="localhost",
        port=5432,
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        """
        DELETE FROM shopping_carts.shopping_cart_items
        WHERE user_id=%s AND shopping_cart_id=%s AND product_id=%s
        """,
        (user_id, cart_id, product_id),
    )

    deleted = cur.rowcount > 0

    cur.close()
    conn.close()

    return deleted