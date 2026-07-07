from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Talent Intelligence System"
    API_V1_STR: str = "/api/v1"
    GEMINI_API_KEY: str
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Weights for Scoring
    WEIGHT_SEMANTIC: float = 0.5
    WEIGHT_EXPERIENCE: float = 0.5

    class Config:
        env_file = ".env"

settings = Settings()