from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Local dev convenience:
    # - `.env` for standard setups
    # - `.env.local` for machine-specific secrets (gitignored via `.env.*`)
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    API_KEY: str

    # RelevanceAI agent config (used by /agent/chat + /agent/poll in app/main.py)
    # Keep optional so the backend can boot without the agent (tools/UI endpoints may still work).
    RAI_API_KEY: str | None = None
    RAI_REGION: str | None = None
    RAI_PROJECT: str | None = None
    RAI_AGENT_ID: str | None = None

    # OpenAI-compatible LLM (used for format_request interpretation)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str | None = None
    OPENAI_BASE_URL: str | None = None


settings = Settings()
