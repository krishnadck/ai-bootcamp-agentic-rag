from pydantic import BaseModel, Field
from typing import Annotated, Any, List, Dict
from langgraph.graph.message import add_messages
from operator import add

class Toolcall(BaseModel):
    name: str = Field(description="Exact tool name from the available tools list")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of ALL required parameters for the tool."
    )
    
class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: List[RAGUsedContext] = Field(description="List of RAG Context items used to answer the question")
    trace_id: str = Field(description="The trace ID of the run", default=None)
        
class AgentProperties(BaseModel):
    final_answer: bool = False
    iteration: int = 0
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[Toolcall] = []
    
class Delegation(BaseModel):
    agent: str
    query: str
    
class CoOrdinatorAgentProperties(BaseModel):
    final_answer: bool = False
    iteration: int = 0
    plan: List[Delegation] = []
    next_agent: str = ""

class CoOrdinatorAgentResponse(BaseModel):
    next_agent: str
    final_answer: bool = False
    answer: str = ""
    plan: List[Delegation] = []
    
class ProductQnAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.", default="")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")
    final_answer: bool = False
    tool_calls: List[Toolcall] = []

class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.", default="")
    final_answer: bool = False
    tool_calls: List[Toolcall] = []

class State(BaseModel):
    messages: Annotated[List[Any], add_messages] = []
    user_intent: str = ""
    product_qna_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    warehouse_manager_agent: AgentProperties = Field(default_factory=AgentProperties)
    co_ordinator_agent: CoOrdinatorAgentProperties = Field(default_factory=CoOrdinatorAgentProperties)
    answer: str = ""
    user_id: str = ""
    cart_id: str = ""
    references: List[RAGUsedContext] = []

class WarehouseManagerAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[Toolcall] = []
