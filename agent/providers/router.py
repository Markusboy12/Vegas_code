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
    
    Распределение ролей:
    - plan: декомпозиция задач (OpenRouter Nemotron)
    - generate: быстрая генерация кода (Groq qwen3.6-27B)
    - fix_early: первые попытки исправления (Groq qwen3.6-27B)
    - fix_late: эскалация при залипании (Groq gpt-oss-120B / OrcaRouter)
    - fix_deep: глубокий разбор traceback (OpenRouter Nemotron)
    - critic: код-ревью (Agnes Flash)
    - memory: извлечение фактов (любой доступный)
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
