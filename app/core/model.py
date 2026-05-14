"""
Model factory for flexible LLM provider switching.

Supports:
- anthropic (Claude via langchain-anthropic)
- google (Gemini via langchain-google-genai)
- minimax (Minimax via langchain-anthropic with custom base_url)

Configure via MODEL_PROVIDER env var. Per-provider API key dicek lazy
saat factory dipanggil (bukan saat import) — supaya gak fail kalau key
provider lain belum di-set.
"""

from typing import Any, Literal, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from app.config import settings


ModelProvider = Literal["anthropic", "google", "minimax"]


def extract_reply(ai_msg: Any) -> Tuple[str, str]:
    """
    Pisahin user-facing reply + reasoning dari AIMessage.

    Dispatch by shape (bukan by provider):
    - content string → reply apa adanya, reasoning kosong.
    - content list-of-blocks → iterate, gabung text-block jadi reply,
      gabung thinking-block jadi reasoning.

    Format list-block dipakai Minimax M-series + Claude extended-thinking.
    Provider lain (Gemini, Claude tanpa thinking) tetep return string.

    Returns: (reply_text, reasoning_text). reasoning_text="" kalau gak ada.
    """
    content = ai_msg.content

    if isinstance(content, str):
        return content, ""

    reply_parts = []
    reasoning_parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            reply_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning_parts.append(block.get("thinking", ""))

    return "".join(reply_parts), "\n".join(reasoning_parts)


def create_llm(
    provider: Optional[ModelProvider] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """
    Create main LLM instance (used for reply generation).

    Args:
        provider: Override settings.MODEL_PROVIDER. None = use env default.
        temperature: Override default reply temperature. None = use settings.REPLY_TEMPERATURE.

    Returns:
        BaseChatModel — provider-agnostic LangChain interface.

    Raises:
        ValueError: kalau provider unknown atau API key untuk provider terpilih kosong.
    """
    provider = provider or settings.MODEL_PROVIDER
    temp = temperature if temperature is not None else settings.REPLY_TEMPERATURE

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY tidak di-set di .env (MODEL_PROVIDER=anthropic)."
            )
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temp,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY tidak di-set di .env (MODEL_PROVIDER=google)."
            )
        if not settings.MODEL_NAME:
            raise ValueError(
                "MODEL_NAME tidak di-set di .env (MODEL_PROVIDER=google)."
            )
        return ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=temp,
        )

    elif provider == "minimax":
        from langchain_anthropic import ChatAnthropic

        if not settings.MINIMAX_API_KEY:
            raise ValueError(
                "MINIMAX_API_KEY tidak di-set di .env (MODEL_PROVIDER=minimax)."
            )
        return ChatAnthropic(
            base_url=settings.MINIMAX_BASE_URL,
            api_key=settings.MINIMAX_API_KEY,
            model=settings.MINIMAX_MODEL,
            temperature=temp,
        )

    else:
        raise ValueError(
            f"Unsupported MODEL_PROVIDER: {provider!r}. "
            "Pilih salah satu: anthropic | google | minimax."
        )


def create_fast_llm(
    provider: Optional[ModelProvider] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """
    Fast/cheap model untuk task ringan (routing, classification, structured output).
    Dipakai unified processor.

    Anthropic → Haiku, Google → Flash. Minimax fallback ke main model
    (Minimax saat ini single-tier).
    """
    provider = provider or settings.MODEL_PROVIDER
    temp = temperature if temperature is not None else settings.UNIFIED_PROCESSOR_TEMPERATURE

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY tidak di-set di .env (MODEL_PROVIDER=anthropic)."
            )
        return ChatAnthropic(
            model=settings.ANTHROPIC_FAST_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temp,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY tidak di-set di .env (MODEL_PROVIDER=google)."
            )
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_FAST_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=temp,
        )

    else:
        # minimax / unknown → fallback ke main LLM
        return create_llm(provider, temperature=temp)
