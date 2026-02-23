from qdrant_client import QdrantClient
from qdrant_client.models import Document, Prefetch, FusionQuery, Filter, FieldCondition, MatchAny, MatchValue
import openai
from langsmith import traceable, get_current_run_tree
from typing import List
from server.agents.reranker import get_reranker
from server.core.config import config
from typing import Dict, Any

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

def fetch_formatted_reviews_context(query: str, item_list: list, top_k: int = 15) -> str:
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

### Shopping Cart Tools

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
            host="postgres",
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
        host="postgres",
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
        host="postgres",
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

@traceable(name="check_warehouse_availability",
           description="Check availability of items across warehouses",
           run_type="tool")
def check_warehouse_availability(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check availability of items across warehouses, including partial fulfillment options.

    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.

    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor

    PG_CONN_INFO = {
        "dbname": "tools_database",
        "user": "langgraph_user",
        "password": "langgraph_password",
        "host": "postgres",
        "port": 5432,
    }

    try:
        conn = psycopg2.connect(**PG_CONN_INFO)
    except Exception as e:
        return {"error": f"Database connection failed: {e}"}

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            all_product_ids = [item["product_id"] for item in items]
            cur.execute(
                """
                SELECT warehouse_id, product_id, available_quantity, warehouse_location, warehouse_name
                FROM warehouses.inventory
                WHERE product_id = ANY(%s)
                """,
                (all_product_ids,),
            )
            inventory_rows = cur.fetchall()
    except Exception as e:
        conn.close()
        return {"error": f"Inventory query failed: {e}"}

    warehouse_map: Dict[str, Any] = {}
    for row in inventory_rows:
        wid = row["warehouse_id"]
        pid = row["product_id"]
        avail = row["available_quantity"]
        loc = row.get("warehouse_location")
        name = row.get("warehouse_name")
        if wid not in warehouse_map:
            warehouse_map[wid] = {
                "items": {},
                "warehouse_location": loc,
                "warehouse_name": name,
            }
        warehouse_map[wid]["items"][pid] = avail

    warehouses_full_fulfillment = []
    warehouses_partial_fulfillment = []
    details: Dict[str, Any] = {}
    unavailable_items = []
    fully_unavailable_products: set = set()

    for wid, wdata in warehouse_map.items():
        can_fulfill_all = True
        partial_fulfillment = False
        warehouse_detail = {
            "warehouse_id": wid,
            "warehouse_location": wdata["warehouse_location"],
            "warehouse_name": wdata["warehouse_name"],
            "items": [],
        }

        for item in items:
            pid = item["product_id"]
            qty = item["quantity"]
            available = wdata["items"].get(pid, 0)

            if available >= qty:
                status = "available"
            elif available > 0:
                status = "partial"
                can_fulfill_all = False
                partial_fulfillment = True
            else:
                status = "unavailable"
                available = 0
                can_fulfill_all = False
                partial_fulfillment = True

            warehouse_detail["items"].append({
                "product_id": pid,
                "requested_quantity": qty,
                "available_quantity": available,
                "status": status,
            })

        details[wid] = warehouse_detail
        if can_fulfill_all:
            warehouses_full_fulfillment.append(wid)
        elif partial_fulfillment:
            warehouses_partial_fulfillment.append(wid)

    for item in items:
        pid = item["product_id"]
        qty = item["quantity"]
        found = any(
            wdata["items"].get(pid, 0) >= qty for wdata in warehouse_map.values()
        )
        if not found:
            max_anywhere = max(
                (wdata["items"].get(pid, 0) for wdata in warehouse_map.values()),
                default=0,
            )
            if max_anywhere == 0:
                fully_unavailable_products.add(pid)
            unavailable_items.append({
                "product_id": pid,
                "requested_quantity": qty,
                "max_available_quantity": max_anywhere,
                "status": "fully_unavailable" if max_anywhere == 0 else "partially_available",
            })

    conn.close()

    return {
        "can_fulfill_completely": len(warehouses_full_fulfillment) > 0,
        "warehouses_full_fulfillment": warehouses_full_fulfillment,
        "warehouses_partial_fulfillment": warehouses_partial_fulfillment,
        "unavailable_items": unavailable_items,
        "details": details,
    }


@traceable(name="reserve_warehouse_items",
           description="Reserve items from warehouses",
           run_type="tool")
def reserve_warehouse_items(reservations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reserve items from multiple warehouses. Successful reservations are committed
    even if some items fail (savepoint-per-item).

    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
                     - warehouse_id: The warehouse to reserve from
                     - product_id: The product to reserve
                     - quantity: The quantity to reserve

    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved
    """
    import psycopg2

    def get_db_conn():
        return psycopg2.connect(
            dbname="tools_database",
            user="langgraph_user",
            password="langgraph_password",
            host="postgres",
            port=5432,
        )

    reserved_items: list = []
    failed_items: list = []

    conn = None
    try:
        conn = get_db_conn()
        conn.autocommit = False
        cur = conn.cursor()

        for idx, reservation in enumerate(reservations):
            warehouse_id = reservation["warehouse_id"]
            product_id = reservation["product_id"]
            quantity = reservation["quantity"]

            savepoint = f"sp_{idx}"
            cur.execute(f"SAVEPOINT {savepoint}")

            try:
                cur.execute(
                    """
                    SELECT id, available_quantity, reserved_quantity, total_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id=%s AND product_id=%s
                    FOR UPDATE
                    """,
                    (warehouse_id, product_id),
                )
                row = cur.fetchone()

                if row is None:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    failed_items.append({
                        "warehouse_id": warehouse_id,
                        "product_id": product_id,
                        "requested_quantity": quantity,
                        "reason": "Item not found",
                    })
                    continue

                inv_id, available_quantity, reserved_quantity, total_quantity = row

                if available_quantity < quantity:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    failed_items.append({
                        "warehouse_id": warehouse_id,
                        "product_id": product_id,
                        "requested_quantity": quantity,
                        "available_quantity": available_quantity,
                        "reason": "Not enough available quantity",
                    })
                    continue

                cur.execute(
                    """
                    UPDATE warehouses.inventory
                    SET reserved_quantity = reserved_quantity + %s
                    WHERE id=%s
                    RETURNING reserved_quantity, available_quantity
                    """,
                    (quantity, inv_id),
                )
                updated = cur.fetchone()
                cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                reserved_items.append({
                    "warehouse_id": warehouse_id,
                    "product_id": product_id,
                    "reserved_quantity": quantity,
                    "total_reserved": updated[0],
                    "remaining_available": updated[1],
                })

            except Exception as item_err:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                failed_items.append({
                    "warehouse_id": warehouse_id,
                    "product_id": product_id,
                    "requested_quantity": quantity,
                    "reason": str(item_err),
                })

        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        reserved_items = []
        failed_items.append({"error": str(e)})
    finally:
        if conn:
            conn.close()

    return {
        "success": len(failed_items) == 0,
        "reserved_items": reserved_items,
        "failed_items": failed_items,
    }