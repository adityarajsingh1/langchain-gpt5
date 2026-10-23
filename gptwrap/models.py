from langchain_openai import ChatOpenAI

from gptwrap.config import load_settings


def chat_model(settings=None, **overrides):
    """The ChatOpenAI model everything else in this project talks to."""
    settings = settings or load_settings()
    kwargs = dict(
        model=settings.model,
        api_key=settings.api_key,
        timeout=settings.timeout,
        max_retries=2,
    )
    # gpt-5 only accepts the default temperature, so it's not set here at all
    if settings.reasoning_effort:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)
