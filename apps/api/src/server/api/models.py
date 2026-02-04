from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list[dict]

class ChatResponse(BaseModel):
    message: str
    
class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to search the RAG database on amazon products")
    thread_id: str = Field(..., description="The thread id")
    
class RAGUsedContext(BaseModel):
    id: str = Field(..., description="The id of the product used to answer the question")
    description: str = Field(..., description="The short description of the product used to answer the question")
    image_url: str = Field(..., description="The image url of the product used to answer the question")
    price: Optional[float] = Field(..., description="The price of the product used to answer the question")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request id")
    used_context: list[RAGUsedContext] = Field(..., description="used context to answer the question")
    answer: str = Field(..., description="The answer to the question")
    
    