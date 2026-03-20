"""
Central settings loaded from environment / .env file.
Tripletex credentials are NOT stored here — they arrive per-request from the platform.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Google Gemini — free tier: https://aistudio.google.com/app/apikey
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Agent behaviour
    log_folder: str = "./logs"
    dry_run: bool = False


settings = Settings()
