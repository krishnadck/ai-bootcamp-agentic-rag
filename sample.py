e.g :
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
                print(f":warning: Repairing broken history at index {i}: Removing orphaned tool_calls.")
                
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