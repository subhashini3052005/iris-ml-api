from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str
    LOG_LEVEL: str
    MAX_BATCH_SIZE: int
    API_TITLE: str

    class Config:
        env_file = ".env"

settings = Settings()
