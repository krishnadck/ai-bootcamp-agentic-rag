from fastapi import APIRouter, Request, HTTPException
from server.api.models import RAGRequest, RAGResponse
import logging
from server.agents.graph import rag_pipeline_wrapper
from server.api.models import RAGUsedContext

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/")
async def amazon_product_assistant(request: Request, payload: RAGRequest) -> RAGResponse:
    response = rag_pipeline_wrapper(payload.query)
    return RAGResponse(request_id=request.state.request_id, answer=response["answer"], 
                       used_context=[RAGUsedContext(**item) for item in response["used_context"]])

api_router = APIRouter()
api_router.include_router(router, prefix="/product_assistant", tags=["rag"])
