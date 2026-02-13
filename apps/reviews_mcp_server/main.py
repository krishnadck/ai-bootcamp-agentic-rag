from math import prod
from fastmcp import FastMCP
from tools import get_formatted_reviews_context

mcp = FastMCP("reviews-mcp-server")

@mcp.tool()
def get_formatted_reviews(query: str, product_ids: list) -> str:
    """Retrieve top matching reviews for a list of product IDs and return formatted reviews context.
    Args:
        query: User search query describing the desired reviews.
        product_ids: A list of product IDs (parent_asin) for which reviews are to be retrieved. 
        Returns:
        Formatted reviews context entries from the review retriever.
    """
    formatted_reviews = get_formatted_reviews_context(query=query, item_list=product_ids)
    
    return formatted_reviews

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)