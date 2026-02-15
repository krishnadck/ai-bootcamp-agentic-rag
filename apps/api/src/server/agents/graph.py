from server.agents.models import State
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from server.agents.tools import retrieve_products, get_formatted_reviews_context
from server.agents.agents import router_node, query_rewriter_node, agent_node
from typing import Literal
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from server.core.config import config
from langgraph.checkpoint.postgres import PostgresSaver
from server.agents.utils.utils import get_tool_descriptions
import json


# edges and graph definitions

def router_conditional_edge(state: State) -> Literal["query_rewriter", END]:
    """
    This function decides the next node to execute based on the user query
    """
    if state.query_relevant:
        return "query_rewriter"
    else:
        return END

# define custom route edge to decide the tool call or agent node or aggregation node
def custome_route_edge(state: State) -> Literal["ageaggregation_nodent", "tools", END]:
    """
    This function decides the next node to execute based on the user query
    """
    #print(state.messages)
    
    if state.final_answer:
        return "end"
    
    if state.iteration > 2:
        return "end"
    
    if len(state.tool_calls) > 0:
        return "tools"
    
    return "end"


def build_graph():
    graphbuilder2 = StateGraph(State)

    tools_node = ToolNode(tools=[retrieve_products, get_formatted_reviews_context])
    graphbuilder2.add_node("router", router_node)
    graphbuilder2.add_node("query_rewriter", query_rewriter_node)
    graphbuilder2.add_node("agent_node", agent_node)
    graphbuilder2.add_node("tools", tools_node)

    graphbuilder2.add_edge(START, "router")
    graphbuilder2.add_conditional_edges("router", router_conditional_edge, {"query_rewriter": "query_rewriter", END: END})
    graphbuilder2.add_edge("query_rewriter", "agent_node")
    graphbuilder2.add_conditional_edges("agent_node", custome_route_edge, {"tools": "tools", "end": END})
    graphbuilder2.add_edge("tools", "agent_node")

    return graphbuilder2


tools=[retrieve_products, get_formatted_reviews_context]
tool_descriptions = get_tool_descriptions(tools)

def run_agent(question, thread_id):
    
    graph_builder = build_graph()
    
    initial_state = {
    "messages": [question],
    "available_tools": tool_descriptions,
    "iteration": 0,
    "final_answer": False,
    }

    thread_config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    with PostgresSaver.from_conn_string(config.postgres_url) as saver:
        
        graph = graph_builder.compile(checkpointer=saver)
        result = graph.invoke(initial_state, config=thread_config)
    
    return result
    
def rag_pipeline_wrapper(question, thread_id=None):
    
    qdrant_client = QdrantClient(   
        url=config.qdrant_url,
    )
    
    result = run_agent(question, thread_id)
    
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
        "trace_id": result.get("trace_id", None)
    }
    
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
            if tool_name == "get_formatted_reviews_context":
                return "Reading relevant user reviews"
            if tool_name:
                return f"Running tool: {tool_name}"
            return "Running tool"

        def _event_message_for_node(node_name):
            node_messages = {
                "router": "Understanding your request...",
                "query_rewriter": "Refining the search query...",
                "agent_node": "Planning the best response...",
                "tools": "Gathering product and review evidence...",
            }
            return node_messages.get(node_name, False)

        if mode == "debug" and isinstance(payload, dict) and payload.get("type") == "task":
            node_name = payload.get("payload", {}).get("name")
            if node_name == "tools":
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

            if node_name == "agent_node":
                if node_update.get("final_answer"):
                    return "Finalizing the answer..."
                if node_update.get("tool_calls"):
                    return "Selecting tools to gather evidence..."
            if node_name == "tools":
                return "Processing retrieved context..."
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
    "messages": [question],
    "available_tools": tool_descriptions,
    "iteration": 0,
    "final_answer": False,
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
            
    yield _string_for_sse(json.dumps(
        {
            "answer": result.get("answer", ""),
            "used_context": used_context,
            "trace_id": result.get("trace_id", None)
        }))
    yield "data: [DONE]\n\n"
    
    
    
    