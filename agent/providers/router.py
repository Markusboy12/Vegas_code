"""Router для управления вызовами к различным LLM-провайдерам."""

from typing import Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from .config import PROVIDERS

logger = structlog.get_logger(__name__)


class AgentRouter:
    """
    Router для вызова LLM с поддержкой ролей и fallback-цепочек.
    
    Принцип распределения:
    Пять провайдеров распределены по ролям в агентном цикле, где у каждого своя сильная сторона:
    - OrcaRouter: мета-шлюз с Qwen-3.8-27B, DeepSeek V4 Flash/Pro, Hy3
    - Ollama Cloud: Gemma4:31b как резервный fallback
    - OpenRouter: Ox Alpha, Laguna S/XS 2.1 (free), North Mini Code (free), Nemotron 3 Ultra (free)
    - GroqAPI: qwen3.6-27B (быстрый), gpt-oss-120B (мощный)
    - Opencode Zen: Hy3, Ox Alpha, Muse Spark 1.2 Contributor
    
    Fallback-цепочка при отказе:
    Groq → OrcaRouter → OpenRouter (free) → Ollama Cloud → Opencode Zen
    """

    def __init__(self):
        self.clients: dict[str, OpenAI] = {}
        for name, cfg in PROVIDERS.items():
            if cfg["api_key"]:
                self.clients[name] = OpenAI(
                    base_url=cfg["base_url"],
                    api_key=cfg["api_key"]
                )
            else:
                logger.warning(f"Provider {name} not configured (no API key)")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        reraise=True
    )
    def call(
        self,
        role: str,
        messages: list[dict],
        fallback_chain: Optional[list[tuple[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Вызов LLM с указанием роли и fallback-цепочки.
        
        Args:
            role: Логическая роль (generate/plan/critic/fix)
            messages: Список сообщений в формате OpenAI
            fallback_chain: [(provider_name, model_key), ...] в порядке приоритета
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            
        Returns:
            Текст ответа от модели
            
        Raises:
            RuntimeError: Если все провайдеры в цепочке недоступны
        """
        if fallback_chain is None:
            from .config import FALLBACK_CHAIN
            fallback_chain = FALLBACK_CHAIN
        
        last_error: Optional[Exception] = None
        
        for provider_name, model_key in fallback_chain:
            if provider_name not in self.clients:
                logger.warning(f"Provider {provider_name} not available (not configured)")
                continue
                
            try:
                client = self.clients[provider_name]
                model = PROVIDERS[provider_name]["models"].get(model_key)
                
                if not model:
                    logger.warning(f"Model {model_key} not found for provider {provider_name}")
                    continue
                
                logger.info(
                    f"Calling {provider_name}/{model} for role={role}",
                    provider=provider_name,
                    model=model,
                    role=role
                )
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                result = response.choices[0].message.content
                logger.info(f"Successfully got response from {provider_name}/{model}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Provider {provider_name} failed: {e}",
                    provider=provider_name,
                    error=str(e)
                )
                continue
        
        raise RuntimeError(f"All providers unavailable for role {role}: {last_error}")

    async def call_async(
        self,
        role: str,
        messages: list[dict],
        fallback_chain: Optional[list[tuple[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Асинхронная версия call()."""
        # В текущей реализации openai SDK асинхронность достигается через aiohttp
        # Для простоты используем синхронный вызов (можно расширить при необходимости)
        return self.call(role, messages, fallback_chain, temperature, max_tokens)
