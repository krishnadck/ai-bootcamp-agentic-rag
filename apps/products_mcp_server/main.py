from fastmcp import FastMCP
from tools import retrieve_products

mcp = FastMCP("products-mcp-server")

@mcp.tool()
def get_formatted_product_context(query: str) -> list:
    """Retrieve top matching products for a query and return formatted product context.

    Args:
        query: User search query describing the desired product(s).

    Returns:
        Formatted product context entries from the product retriever.
    """

    formatted_retrieved_context = retrieve_products(query)
    
    return formatted_retrieved_context

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)