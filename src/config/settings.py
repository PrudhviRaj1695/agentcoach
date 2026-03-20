"""Application configuration — loads from .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """AgentCoach configuration."""
    
    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-06-01"
    azure_openai_deployment_gpt4: str = ""
    azure_openai_deployment_gpt4_mini: str = ""
    azure_openai_deployment_embedding: str = ""
    
    # App
    app_env: str = "development"
    log_level: str = "INFO"
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()