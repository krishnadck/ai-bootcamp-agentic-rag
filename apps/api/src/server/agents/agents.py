from server.agents.utils.prompt_management import get_prompt_from_config
from langchain_core.messages import ToolMessage
from server.agents.models import State
import instructor
from openai import OpenAI
from langsmith import traceable
from server.agents.utils.utils import format_ai_message
from langchain_core.messages import AIMessage, convert_to_openai_messages
from langsmith import get_current_run_tree
from server.agents.models import ProductQnAAgentResponse, ShoppingCartAgentResponse, CoOrdinatorAgentResponse
from server.agents.utils.utils import normalize_tool_calls


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
def product_qna_agent_node(state: State) -> State:
    """
    This function uses the RAG pipeline to perform search on the products
    """
    template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/search_agent.yml', 'search_agent')
    
    prompt = template.render(available_tools=state.product_qna_agent.available_tools)

    messages = sanitize_history(state.messages)
    
    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))
        
    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=ProductQnAAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )
    
    current_run = get_current_run_tree()
    
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
    
    ai_message = format_ai_message(response)
    
    ai_message = normalize_tool_calls(ai_message)
    

    return {
        "messages": [ai_message],
        "product_qna_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.product_qna_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.product_qna_agent.available_tools
        },
        "answer": response.answer,
        "references": response.references
    }

@traceable(name="shopping_cart_agent_node", 
description="This function uses the Shopping Cart tools to add, get and remove items from the cart.",
run_type="llm"
)
def shopping_cart_agent_node(state: State) -> State:
    """
    This function uses the Shopping Cart tools to add, get and remove items from the cart.
    """
    
    template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/shopping_cart_agent.yml', 'shopping_cart_agent')
    
    prompt = template.render(user_id=state.user_id, cart_id=state.cart_id, available_tools=state.shopping_cart_agent.available_tools)
    
    conversation = []
    for message in state.messages:
        conversation.append(convert_to_openai_messages(message))
        
    client = instructor.from_openai(OpenAI())
    
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=ShoppingCartAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0,
    )
    
    current_run = get_current_run_tree()
    
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
    
    ai_message = format_ai_message(response)
    
    ai_message = normalize_tool_calls(ai_message)
    
    return {
        "messages": [ai_message],
        "shopping_cart_agent": {
            "tool_calls": [tool_call.model_dump() for tool_call in response.tool_calls],
            "iteration": state.shopping_cart_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.shopping_cart_agent.available_tools
        },
        "answer": response.answer
    }

@traceable(name="coordinator_agent_node", 
description="This function evaluates the user query and decides the next node to execute",
run_type="llm"
)
def coordinator_agent_node(state: State) -> State:
    """
    This function evaluates the user query and decides the next node to execute
    """
    prompt_template = get_prompt_from_config('/app/apps/api/src/server/agents/prompts/coordinator_agent.yml', 'coordinator_agent')
    
    prompt = prompt_template.render()
    
    conversation = []
    
    for message in state.messages:
        conversation.append(convert_to_openai_messages(message))
    
    client = instructor.from_openai(OpenAI())
    
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=CoOrdinatorAgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.0
    )
    
    current_run = get_current_run_tree()
    
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))
    else:
        trace_id = None
    
    if response.final_answer:
        ai_message = [AIMessage(content=response.answer)]
    else:
        ai_message = []

    return {
       "messages": ai_message,
       "answer": response.answer,
       "co_ordinator_agent": {
           "iteration": state.co_ordinator_agent.iteration + 1,
           "final_answer": response.final_answer,
           "plan": [plan.model_dump() for plan in response.plan],
           "next_agent": response.next_agent
       },
       "trace_id": trace_id
    }