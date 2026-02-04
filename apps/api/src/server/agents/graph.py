from server.agents.models import State
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from server.agents.tools import retrieve_embedding
from server.agents.agents import router_node, query_rewriter_node, agent_node
from langchain_core.messages import AIMessage
from typing import Literal
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from server.core.config import config
from langgraph.checkpoint.postgres import PostgresSaver
from server.agents.utils.utils import get_tool_descriptions


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

    tools_node = ToolNode(tools=[retrieve_embedding])
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


tools=[retrieve_embedding]
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
    }
    