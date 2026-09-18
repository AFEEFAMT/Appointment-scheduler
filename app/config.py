from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    gemini_api_key: SecretStr = SecretStr("")

    gemini_primary_model: str = "gemini-3.8-flash"
    gemini_fallback_models: str = (
        "gemini-3.5-flash,gemini-2.5-flash"
    )

    app_timezone: str = "Asia/Kolkata"

    max_image_size_mb: int = Field(default=5, ge=1, le=20)
    ocr_confidence_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )

    llm_timeout_seconds: float = Field(default=20, ge=5, le=120)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_base_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10,
    )

    cache_ttl_seconds: int = Field(default=600, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def gemini_model_chain(self) -> tuple[str, ...]:
        """Return primary and fallback models in execution order."""

        fallback_models = [
            model.strip()
            for model in self.gemini_fallback_models.split(",")
            if model.strip()
        ]

        return tuple(
            dict.fromkeys(
                [self.gemini_primary_model, *fallback_models]
            )
        )

    @property
    def gemini_is_configured(self) -> bool:
        """Check whether a non-placeholder API key is configured."""

        api_key = self.gemini_api_key.get_secret_value().strip()

        return bool(
            api_key
            and api_key != "your_key_here"
            and "PASTE_" not in api_key
        )


@lru_cache
def get_settings() -> Settings:
    """Create and cache one settings object."""

    return Settings()


settings = get_settings()