from pydantic import BaseModel, Field
from typing import Annotated, Any, List, Dict
from langgraph.graph.message import add_messages
from operator import add

class Toolcall(BaseModel):
    name: str
    arguments: dict
    
class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class AgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")
    final_answer: bool = False
    tool_calls: List[Toolcall] = []

class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: List[RAGUsedContext] = Field(description="List of RAG Context items used to answer the question")
    
class QueryRelevanceResponse(BaseModel):
    query_relevant: bool
    reason: str

class QueryRewriteResponse(BaseModel):
    search_queries: List[str]
    
class AggregationResponse(BaseModel):
    answer: str = Field(description="The answer to the question in a list format.")
    references: List[RAGUsedContext] = Field(description="List of RAG Context items used to answer the question")
    
class State(BaseModel):
    messages: Annotated[List[Any], add_messages] = []
    expanded_queries: List[str] = []
    final_answer: bool = False
    iteration: int = 0
    available_tools: List[Dict[str, Any]] = []
    answer: str = ""
    query_relevant: bool = False
    tool_calls: List[Toolcall] = []
    references: Annotated[List[RAGUsedContext], add] = []
