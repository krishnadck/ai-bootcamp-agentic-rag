from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    langsmith_api_key: Optional[str] = None
    langsmith_tracing: Optional[str] = None # Or bool
    langsmith_endpoint: Optional[str] = None
    langsmith_project: Optional[str] = None

    openai_api_key: str
    google_api_key: str
    groq_api_key: str
    qdrant_url: str = "http://localhost:6333"
    postgres_url: str = "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"

config = Config()
