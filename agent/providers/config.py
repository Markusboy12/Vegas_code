"""Конфигурация провайдеров с распределением по ролям."""

import os

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "models": {"fast": "qwen3.6-27b", "strong": "gpt-oss-120b"},
    },
    "orcarouter": {
        "base_url": "https://api.orcarouter.ai/v1",
        "api_key": os.environ.get("ORCAROUTER_API_KEY", ""),
        "models": {"auto": "orcarouter/auto", "qwen": "qwen-3.8-27b"},
    },
    "agnes": {
        "base_url": "https://api.agnes-ai.com/v1",
        "api_key": os.environ.get("AGNES_API_KEY", ""),
        "models": {"critic": "agnes-2.5-flash"},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "models": {"planner": "nvidia/nemotron-nano-9b-v1"},
    },
    "ollama_cloud": {
        "base_url": "https://ollama.com/v1",
        "api_key": os.environ.get("OLLAMA_API_KEY", ""),
        "models": {"fallback": "gemma4:31b"},
    },
    "opencode_zen": {
        "base_url": "https://api.opencode-zen.com/v1",
        "api_key": os.environ.get("OPENCODE_ZEN_API_KEY", ""),
        "models": {"fallback": "hy3", "deepseek": "deepseek-coder"},
    },
}

# Распределение ролей в агентном цикле
# Принцип: 6 провайдеров — не просто список альтернатив, а распределение по ролям
ROLE_MAPPING = {
    # Планирование/декомпозиция задач — Nemotron (сильный reasoning)
    "plan": [("openrouter", "planner")],
    
    # Быстрая генерация кода — Groq LPU (критично для итеративного цикла)
    "generate": [("groq", "fast")],
    
    # Первые попытки исправления (1-2) — скорость важна
    "fix_early": [("groq", "fast")],
    
    # Эскалация при залипании (попытка 3+) — крупная модель или OrcaRouter
    "fix_late": [("groq", "strong"), ("orcarouter", "auto")],
    
    # Глубокий разбор traceback — Nemotron reasoning
    "fix_deep": [("openrouter", "planner")],
    
    # Critic/код-ревью — Agnes Flash (другая модельная семья, контекст 256K-512K)
    "critic": [("agnes", "critic")],
    
    # Извлечение фактов для памяти — любой доступный (скорость не критична)
    "memory": [("groq", "fast")],
}

# Fallback-цепочка при отказе провайдера (мета-шлюз)
# Groq → OrcaRouter → Ollama Cloud → Opencode Zen
FALLBACK_CHAIN = [
    ("groq", "fast"),
    ("orcarouter", "auto"),
    ("ollama_cloud", "fallback"),
    ("opencode_zen", "fallback"),
]
