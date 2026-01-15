from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    API_URL: str = "http://api:8000/product_assistant"
    
config = Config()
