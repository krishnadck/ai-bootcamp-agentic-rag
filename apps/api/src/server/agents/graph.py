from server.agents.models import State
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from server.agents.tools import retrieve_embedding
from server.agents.agents import router_node, query_rewriter_node, agent_node, aggregation_node
from langchain_core.messages import AIMessage
from typing import Literal
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from server.core.config import config

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
    
    tool_calls_count = 0
    for m in state.messages:
        if m.tool_calls and isinstance(m, AIMessage):
            tool_calls_count += len(m.tool_calls)
    
    print(f"tool_calls_count: {tool_calls_count}")
    
    if tool_calls_count == 0:
        return "aggregation"
    
    if tools_condition(state.messages) == "tools" and tool_calls_count <= len(state.expanded_queries):
        return "tools"
    
    return "aggregation"

def build_graph():
    graphbuilder2 = StateGraph(State)

    tools_node = ToolNode(tools=[retrieve_embedding])
    graphbuilder2.add_node("router", router_node)
    graphbuilder2.add_node("query_rewriter", query_rewriter_node)
    graphbuilder2.add_node("agent_node", agent_node)
    graphbuilder2.add_node("aggregation", aggregation_node)
    graphbuilder2.add_node("tools", tools_node)

    graphbuilder2.add_edge(START, "router")
    graphbuilder2.add_conditional_edges("router", router_conditional_edge, {"query_rewriter": "query_rewriter", END: END})
    graphbuilder2.add_edge("query_rewriter", "agent_node")
    graphbuilder2.add_conditional_edges("agent_node", custome_route_edge, {"tools": "tools", "aggregation": "aggregation", END: "aggregation"})
    graphbuilder2.add_edge("tools", "aggregation")
    graphbuilder2.add_edge("aggregation", END)

    agg_graph_1 = graphbuilder2.compile()
    return agg_graph_1


def rag_pipeline_wrapper(question, top_k=10):
    
    qdrant_client = QdrantClient(   
        url=config.qdrant_url,
    )
    
    graph = build_graph()
    
    initial_state = State(user_query=question)
    
    result = graph.invoke(initial_state)
            
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
    