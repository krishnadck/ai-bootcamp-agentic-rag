from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from server.api.models import RAGRequest
import logging
from server.agents.graph import rag_agent_stream_wrapper

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/")
async def amazon_product_assistant(request: Request, payload: RAGRequest) -> StreamingResponse:
    logger.info(f"Received request: {payload.query} with thread_id: {payload.thread_id}")
    return StreamingResponse(
        rag_agent_stream_wrapper(payload.query, thread_id=payload.thread_id), media_type="text/event-stream")


api_router = APIRouter()
api_router.include_router(router, prefix="/product_assistant", tags=["agent"])
