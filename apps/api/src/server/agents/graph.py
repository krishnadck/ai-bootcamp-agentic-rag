from server.agents.models import State
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from server.agents.tools import retrieve_products, fetch_formatted_reviews_context
from server.agents.tools import add_to_shopping_cart, read_shopping_cart, remove_item_from_cart, check_warehouse_availability
from server.agents.tools import reserve_warehouse_items
from server.agents.agents import product_qna_agent_node, shopping_cart_agent_node, coordinator_agent_node, warehouse_manager_agent_node
from typing import Literal
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from server.core.config import config
from langgraph.checkpoint.postgres import PostgresSaver
from server.agents.utils.utils import get_tool_descriptions
import json

# edges and graph definitions

def product_qa_router_edge(state: State) -> Literal["tools", "coordinator_agent"]:
    """
    This function decides the next node to execute based on the user query
    """
    if state.product_qna_agent.iteration > 4:
        return "coordinator_agent"
    elif state.product_qna_agent.final_answer and len(state.co_ordinator_agent.plan) == 0:
        return "coordinator_agent"
    elif len(state.product_qna_agent.tool_calls) > 0:
        return "tools"
    else:
        return "coordinator_agent"

def shopping_cart_router_edge(state: State) -> Literal["tools", "coordinator_agent"]:
    """ 
    This function decides the next node to execute based on the user query
    """
    
    if state.shopping_cart_agent.iteration > 2:
        return "coordinator_agent"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        return "tools"
    if state.shopping_cart_agent.final_answer:
        return "coordinator_agent"
    else:
        return "coordinator_agent"

def warehouse_manager_router_edge(state: State) -> Literal["tools", "coordinator_agent"]:
    """
    Routes warehouse manager agent output to tools or back to coordinator.
    """
    if state.warehouse_manager_agent.iteration > 3:
        return "coordinator_agent"
    elif len(state.warehouse_manager_agent.tool_calls) > 0:
        return "tools"
    if state.warehouse_manager_agent.final_answer:
        return "coordinator_agent"
    else:
        return "coordinator_agent"
    
def coOrdinator_agent_router_edge(state: State) -> Literal["product_qna_agent", "shopping_cart_agent", "warehouse_manager_agent", END]:
    """
    This function decides the next node to execute based on the user query
    """
    if state.co_ordinator_agent.final_answer:
        return END
    elif state.co_ordinator_agent.iteration > 3:
        return END
    elif state.co_ordinator_agent.next_agent == "product_qa_agent":
        return "product_qna_agent"
    elif state.co_ordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    elif state.co_ordinator_agent.next_agent == "warehouse_manager_agent":
        return "warehouse_manager_agent"
    else:
        return END

product_qna_agent_tools=[retrieve_products, fetch_formatted_reviews_context]
product_qna_agent_tools_descriptions = get_tool_descriptions(product_qna_agent_tools)

shopping_cart_agent_tools=[add_to_shopping_cart, remove_item_from_cart, read_shopping_cart]
shopping_cart_tool_descriptions = get_tool_descriptions(shopping_cart_agent_tools)

warehouse_manager_agent_tools = [check_warehouse_availability, reserve_warehouse_items]
warehouse_manager_tool_descriptions = get_tool_descriptions(warehouse_manager_agent_tools)

    
def build_graph():
    graphbuilder2 = StateGraph(State)

    qna_tools_node = ToolNode(tools=product_qna_agent_tools)
    shopping_cart_tools_node = ToolNode(tools=shopping_cart_agent_tools)   
    warehouse_manager_tools_node = ToolNode(tools=warehouse_manager_agent_tools)

    graphbuilder2.add_node("coordinator_agent", coordinator_agent_node)
    graphbuilder2.add_node("product_qna_agent", product_qna_agent_node)
    graphbuilder2.add_node("shopping_cart_agent", shopping_cart_agent_node)
    graphbuilder2.add_node("warehouse_manager_agent", warehouse_manager_agent_node)

    graphbuilder2.add_node("product_qna_tools", qna_tools_node)
    graphbuilder2.add_node("shopping_cart_tools", shopping_cart_tools_node)
    graphbuilder2.add_node("warehouse_manager_tools", warehouse_manager_tools_node)

    graphbuilder2.add_edge(START, "coordinator_agent")

    graphbuilder2.add_conditional_edges("coordinator_agent", coOrdinator_agent_router_edge, {"product_qna_agent": "product_qna_agent", "shopping_cart_agent": "shopping_cart_agent", "warehouse_manager_agent": "warehouse_manager_agent", END: END})
    graphbuilder2.add_conditional_edges("product_qna_agent", product_qa_router_edge, {"tools": "product_qna_tools", "coordinator_agent": "coordinator_agent"})
    graphbuilder2.add_conditional_edges("shopping_cart_agent", shopping_cart_router_edge, {"tools": "shopping_cart_tools", "coordinator_agent": "coordinator_agent"})
    graphbuilder2.add_conditional_edges("warehouse_manager_agent", warehouse_manager_router_edge, {"tools": "warehouse_manager_tools", "coordinator_agent": "coordinator_agent"})

    graphbuilder2.add_edge("product_qna_tools", "product_qna_agent")
    graphbuilder2.add_edge("shopping_cart_tools", "shopping_cart_agent")
    graphbuilder2.add_edge("warehouse_manager_tools", "warehouse_manager_agent")
    return graphbuilder2

    
def _process_graph_event(chunk):
        mode, payload = chunk

        def _tool_to_text(tool_call):
            tool_name = getattr(tool_call, "name", None)
            arguments = getattr(tool_call, "arguments", {}) or {}

            if tool_name is None and isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {}) or {}

            if tool_name == "retrieve_products":
                search_query = arguments.get("query", "")
                if search_query:
                    return f"Searching products for: {search_query}"
                return "Searching relevant products"
            if tool_name == "fetch_formatted_reviews_context":
                return "Reading relevant user reviews"
            if tool_name == "add_to_shopping_cart":
                return "Updating your shopping cart"
            if tool_name == "read_shopping_cart":
                return "Fetching your current cart"
            if tool_name == "remove_item_from_cart":
                return "Removing item from your cart"
            if tool_name:
                return f"Running tool: {tool_name}"
            return "Running tool"

        def _event_message_for_node(node_name):
            node_messages = {
                "coordinator_agent": "Planning next best step...",
                "product_qna_agent": "Analyzing products and drafting response...",
                "shopping_cart_agent": "Processing your cart request...",
                "product_qna_tools": "Gathering product and review evidence...",
                "shopping_cart_tools": "Applying shopping cart actions...",
            }
            return node_messages.get(node_name, False)

        if mode == "debug" and isinstance(payload, dict) and payload.get("type") == "task":
            node_name = payload.get("payload", {}).get("name")
            if node_name in {"product_qna_tools", "shopping_cart_tools"}:
                input_payload = payload.get("payload", {}).get("input", {})
                tool_calls = getattr(input_payload, "tool_calls", None)
                if tool_calls is None and isinstance(input_payload, dict):
                    tool_calls = input_payload.get("tool_calls", [])
                if tool_calls:
                    return " | ".join(_tool_to_text(tool_call) for tool_call in tool_calls)
            return _event_message_for_node(node_name)

        if mode == "updates" and isinstance(payload, dict) and payload:
            node_name = next(iter(payload.keys()))
            node_update = payload.get(node_name, {}) or {}

            if node_name in {"coordinator_agent", "product_qna_agent", "shopping_cart_agent"}:
                if node_update.get("final_answer"):
                    if node_name == "coordinator_agent":
                        return "Finalizing your response..."
                    return "Sending result back to coordinator..."
                if node_update.get("tool_calls"):
                    return "Selecting tools to gather evidence..."
            if node_name in {"product_qna_tools", "shopping_cart_tools"}:
                return "Processing tool output..."
            return _event_message_for_node(node_name)

        return False

def rag_agent_stream_wrapper(question, thread_id=None):
    
    def _string_for_sse(message: str) -> str:
        return f"data: {message}\n\n"
    
    qdrant_client = QdrantClient(   
        url=config.qdrant_url,
    )
    
    graph_builder = build_graph()
    
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "user_id": "krishnak",
        "cart_id": thread_id,
        "product_qna_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": product_qna_agent_tools_descriptions,
            "tool_calls": []
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": shopping_cart_tool_descriptions,
            "tool_calls": []
        },
        "warehouse_manager_agent": {
        "iteration": 0,
        "final_answer": False,
        "available_tools": warehouse_manager_tool_descriptions,
        "tool_calls": []
        },
        "co_ordinator_agent": {
            "iteration": 0,
            "final_answer": False,
            "plan": [],
            "next_agent": ""
        }
    }
    
    thread_config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

    result = {}
    with PostgresSaver.from_conn_string(config.postgres_url) as saver:
        
        graph = graph_builder.compile(checkpointer=saver)
        
        for chunk in graph.stream(
            initial_state,
            config=thread_config,
            stream_mode=["updates", "debug", "values"]
        ):
            processed_chunk = _process_graph_event(chunk)
            
            if processed_chunk:
                yield _string_for_sse(processed_chunk)
            
            if chunk[0] == "values":
                result = chunk[1]
    
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
    
    shopping_cart = read_shopping_cart("krishnak", thread_id)
    shopping_cart_items = [
        {
            "price": float(item.get("price")) if item.get("price") else None,
            "quantity": item.get("quantity"),
            "currency": item.get("currency"),
            "product_image_url": item.get("product_image_url"),
            "total_price": float(item.get("total_price")) if item.get("total_price") else None
        }
        for item in shopping_cart
    ]
    
    yield _string_for_sse(json.dumps(
        {
            "answer": result.get("answer", ""),
            "used_context": used_context,
            "shopping_cart_items": shopping_cart_items,
            "trace_id": result.get("trace_id", None)
        }))
    yield "data: [DONE]\n\n"
    
    
    
    