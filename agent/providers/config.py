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
ROLE_MAPPING = {
    "plan": [("openrouter", "planner")],
    "generate": [("groq", "fast")],
    "fix_early": [("groq", "fast")],
    "fix_late": [("groq", "strong"), ("orcarouter", "auto")],
    "fix_deep": [("openrouter", "planner")],
    "critic": [("agnes", "critic")],
    "memory": [("groq", "fast")],
}

# Fallback-цепочка при отказе провайдера
FALLBACK_CHAIN = [
    ("groq", "fast"),
    ("orcarouter", "auto"),
    ("ollama_cloud", "fallback"),
    ("opencode_zen", "fallback"),
]
