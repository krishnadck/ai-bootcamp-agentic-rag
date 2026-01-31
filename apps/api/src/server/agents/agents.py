from server.agents.tools import retrieve_embedding
from server.agents.models import AggregationResponse, QueryRewriteResponse, QueryRelevanceResponse
from server.agents.utils.prompt_management import get_prompt_from_config
from langchain_core.messages import ToolMessage
from server.agents.models import State
import instructor
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langsmith import traceable


@traceable(name="aggregation_node", 
description="This function aggregates the retrieved context data and returns the final answer",
run_type="llm"
)
def aggregation_node(state: State) -> State:
    """
    This function aggregates the retrieved context data and returns the final answer
    """
    
    # Extract the formatted product descriptions from state.messages
    product_descriptions = []
    
    for msg in state.messages:
      if isinstance(msg, ToolMessage):
        if msg.artifact:
            for item in msg.artifact:
              product_descriptions.append(item)
      
    template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/aggregator_agent.yml', 
                                      'aggregator_agent')
      
    prompt = template.render(
                    question=state.user_query, 
                    expanded_queries=state.expanded_queries, 
                    preprocessed_context=product_descriptions)
    
    client = instructor.from_openai(OpenAI())
    
    response, raw_response = client.chat.completions.create_with_completion(
    model="gpt-4.1-mini",
    messages=[{"role": "system", "content": prompt}],
    response_model=AggregationResponse,
    temperature=0.4
    )
    
    return {
      "answer": response.answer,
      "references": response.references
    }
    

@traceable(name="query_rewriter_node", 
description="This function rewrites the query to be more specific to include multiple statements",
run_type="prompt"
)
def query_rewriter_node(state: State) -> str:
    """
    This function rewrites the query to be more specific to include multiple statements
    """
    template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/query_expand_agent.yml', 
                                      'query_expand_agent')
    
    prompt = template.render(query=state.user_query)
    
    client = instructor.from_openai(OpenAI())
    
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=QueryRewriteResponse,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4
    )
    return {
        "expanded_queries": response.search_queries
    }
    
# add router node to evaluate the user query and decide the next node to execute
def router_node(state: State) -> State:
    """
    This function evaluates the user query and decides the next node to execute
    """
    
    template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/router_agent.yml', 'router_agent')
    
    prompt = template.render(question=state.user_query)
    
    client = instructor.from_openai(OpenAI())
    
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=QueryRelevanceResponse,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4
    )
    
    return {
        "query_relevant": response.query_relevant,
        "answer": response.reason
    }

@traceable(name="agent_node", 
description="This function uses the RAG pipeline to perform search on the products",
run_type="llm"
)
def agent_node(state: State) -> State:
    """
    This function uses the RAG pipeline to perform search on the products
    """
    template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/search_agent.yml', 'search_agent')
    
    prompt = template.render(expanded_queries=state.expanded_queries)
        
    client = ChatOpenAI(model="gpt-4o-mini", temperature=0.4).bind_tools([retrieve_embedding], tool_choice="required")
    
    response = client.invoke(prompt)
    
    return {
        "messages": [response]
    }