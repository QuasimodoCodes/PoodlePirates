"""
Central settings loaded from environment / .env file.
Tripletex credentials are NOT stored here — they arrive per-request from the platform.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic — the only secret we own
    anthropic_api_key: str
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Agent behaviour
    log_folder: str = "./logs"
    dry_run: bool = False


settings = Settings()
