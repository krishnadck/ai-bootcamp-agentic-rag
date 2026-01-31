from pydantic import BaseModel, Field
from typing import Annotated, Any, List
from langgraph.graph.message import add_messages

class RAGContext(BaseModel):
    id: str = Field(description="The id of the product used to answer the question")
    description: str = Field(description="The short description of the product used to answer the question")

class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: List[RAGContext] = Field(description="List of RAG Context items used to answer the question")
    
class QueryRelevanceResponse(BaseModel):
    query_relevant: bool
    reason: str

class QueryRewriteResponse(BaseModel):
    search_queries: List[str]
    
class AggregationResponse(BaseModel):
    answer: str = Field(description="The answer to the question in a list format.")
    references: List[RAGContext] = Field(description="List of RAG Context items used to answer the question")
    
class State(BaseModel):
    messages: Annotated[List[Any], add_messages] = []
    user_query: str
    expanded_queries: List[str] = []
    answer: str = ""
    references: List[RAGContext] = Field(default_factory=list, description="List of RAG Context items used to answer the question")
    query_relevant: bool = False