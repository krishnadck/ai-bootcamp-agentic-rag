import os
from langsmith import Client


# Ensure your LANGCHAIN_API_KEY is set in your environment variables
client = Client()

def submit_feedback(
    trace_id: str, 
    feedback_score: int | None = None, 
    feedback_text: str = "", 
    feedback_source_type: str = "api"
):
    """
    Submits feedback to LangSmith for a specific run (trace_id).
    
    Args:
        trace_id: The UUID of the run/trace.
        feedback_score: Integer 1 (Like) or 0 (Dislike).
        feedback_text: Optional comment from the user.
        feedback_source_type: Source metadata (e.g., 'web', 'mobile').
    """
    
    # Corrected Logic: Explicitly check for None so '0' is processed
    if feedback_score is not None:
        client.create_feedback(
            run_id=trace_id,
            key="thumbs",  # Standard key for binary feedback
            score=feedback_score,
            feedback_source_type=feedback_source_type
        )

    # Only send comment if it exists
    if feedback_text and len(feedback_text.strip()) > 0:
        client.create_feedback(
            run_id=trace_id,
            key="comment", # Standard key for text feedback
            value=feedback_text,
            feedback_source_type=feedback_source_type
        )