from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./cv_platform.sqlite3"
    storage_root: str = "storage"
    temp_root: str = "temp"
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    auth_jwt_secret: str = "change-this-development-secret-before-deployment"
    auth_access_token_minutes: int = 30
    auth_refresh_token_days: int = 14
    auth_allow_legacy_account_claim: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
