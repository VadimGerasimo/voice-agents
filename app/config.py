from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str

    # ElevenLabs Configuration
    elevenlabs_api_key: str

    # Supabase Configuration
    supabase_url: str
    supabase_key: str

    # Application Configuration
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
