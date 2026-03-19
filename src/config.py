"""
Central settings loaded from environment / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Tripletex
    tripletex_consumer_token: str
    tripletex_employee_token: str
    tripletex_company_id: int = 0
    tripletex_base_url: str = "https://tripletex.no/v2"

    # Anthropic
    anthropic_api_key: str
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Email
    email_host: str = "imap.gmail.com"
    email_port: int = 993
    email_user: str = ""
    email_password: str = ""
    email_folder: str = "INBOX"

    # Agent
    inbox_folder: str = "./inbox"
    log_folder: str = "./logs"
    dry_run: bool = False


settings = Settings()
