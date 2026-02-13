from pydantic import BaseModel, Field, UUID4
from typing import Optional
from server.api.domains.feedback import submit_feedback
from fastapi import HTTPException
from fastapi import APIRouter
import logging

router1 = APIRouter()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pydantic Schema
class FeedbackCreate(BaseModel):
    # UUID4 ensures the ID format is valid before it hits your logic
    run_id: str = Field(str, description="The LangSmith Run/Trace ID")
    
    # Using int to support binary rating (1=Up, 0=Down)
    score: Optional[int] = Field(None, description="1 for Thumbs Up, 0 for Thumbs Down")
    
    # Optional comment
    comment: Optional[str] = Field(None, description="Optional text feedback")
    
    # Metadata for where this came from
    source: str = Field("web", description="Source: web, mobile, etc.")

# --- API Endpoint ---
@router1.post("/", status_code=201)
async def create_feedback(payload: FeedbackCreate):
    """
    Receives user feedback and logs it to LangSmith.
    """
    logger.info(f"Received feedback: {payload}")
    
    try:
        # Map Pydantic fields to your function arguments
        submit_feedback(
            trace_id=str(payload.run_id),
            feedback_score=payload.score,
            feedback_text=payload.comment or "",
            feedback_source_type=payload.source
        )
        return {"status": "success", "message": "Feedback recorded"}
    
    except Exception as e:
        # Log the error internally (e.g., print(e) or logger.error(e))
        # Do not expose the exact error to the client
        print(f"Feedback Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")
    
feedback_router = APIRouter()
feedback_router.include_router(router1, prefix="/feedback", tags=["feedback"])