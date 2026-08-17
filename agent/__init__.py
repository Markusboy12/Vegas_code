"""
Coding AI Agent - агент для генерации и исправления кода.

Архитектура основана на распределении ролей между 6 LLM-провайдерами:
- Groq: быстрая генерация кода (qwen3.6-27B) и эскалация (gpt-oss-120B)
- OrcaRouter: адаптивная маршрутизация с failover
- Agnes AI: код-ревью (Critic)
- OpenRouter: планирование и reasoning (Nemotron)
- Ollama Cloud: резервный fallback
- Opencode Zen: резервный fallback + DeepSeek для кода
"""

from .core.agent import CodeAgent
from .providers.router import AgentRouter
from .sandbox.executor import Sandbox
from .memory.layer import MemoryLayer

__version__ = "1.0.0"
__all__ = ["CodeAgent", "AgentRouter", "Sandbox", "MemoryLayer"]
