"""Модуль провайдеров LLM."""

from .config import PROVIDERS, ROLE_MAPPING, FALLBACK_CHAIN
from .router import AgentRouter

__all__ = ["PROVIDERS", "ROLE_MAPPING", "FALLBACK_CHAIN", "AgentRouter"]
