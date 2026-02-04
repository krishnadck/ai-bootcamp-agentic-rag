from server.agents.tools import retrieve_embedding
from server.agents.models import QueryRewriteResponse, QueryRelevanceResponse
from server.agents.utils.prompt_management import get_prompt_from_config
from langchain_core.messages import ToolMessage
from server.agents.models import State
import instructor
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langsmith import traceable
from server.agents.utils.utils import format_ai_message
from server.agents.models import AgentResponse
from langchain_core.messages import AIMessage, convert_to_openai_messages

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
    
    prompt = template.render(query=state.messages[-1].content)
    
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
    
    prompt = template.render(question=state.messages[-1].content)
    
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

def sanitize_history(messages):
    """
    Scans the entire message history. 
    If an AIMessage has tool_calls but is NOT followed by a ToolMessage,
    we strip the tool_calls from it to prevent OpenAI 400 errors.
    """
    sanitized_msgs = []
    
    # Iterate through all messages
    for i, msg in enumerate(messages):
        
        # Check if this is an AI Message with tool calls
        if isinstance(msg, AIMessage) and msg.tool_calls:
            
            # Look ahead: Is the NEXT message a ToolMessage?
            is_valid_chain = False
            if i + 1 < len(messages):
                next_msg = messages[i+1]
                if isinstance(next_msg, ToolMessage):
                    is_valid_chain = True
            
            if is_valid_chain:
                # It's a valid pair (AI -> Tool). Keep it.
                sanitized_msgs.append(msg)
            else:
                # It's BROKEN (AI -> Human or AI -> End). 
                # Fix: Create a clean copy WITHOUT tool_calls.
                print(f"⚠️ Repairing broken history at index {i}: Removing orphaned tool_calls.")
                
                # We keep the text content (if any), but remove the toxic tool_calls
                clean_msg = msg.model_copy(update={"tool_calls": [], "id": msg.id})
                
                # Only add it if it actually has text (otherwise it's an empty message)
                if clean_msg.content:
                    sanitized_msgs.append(clean_msg)
        
        # If it's a ToolMessage that was orphaned (no previous AI call), 
        # OpenAI usually tolerates this, or you can filter it too. 
        # For now, we just pass non-AI messages through.
        else:
            sanitized_msgs.append(msg)
            
    return sanitized_msgs

@traceable(name="agent_node", 
description="This function uses the RAG pipeline to perform search on the products",
run_type="llm"
)
def agent_node(state: State) -> State:
    """
    This function uses the RAG pipeline to perform search on the products
    """
    template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/search_agent.yml', 'search_agent')
    
    prompt = template.render(available_tools=state.available_tools)

    messages = sanitize_history(state.messages)

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))
        
    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )
    
    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references
    }

