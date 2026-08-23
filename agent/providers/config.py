"""Конфигурация провайдеров с распределением по ролям."""

import os

PROVIDERS = {
    "orcarouter": {
        "base_url": "https://api.orcarouter.ai/v1",
        "api_key": os.environ.get("ORCAROUTER_API_KEY", ""),
        "models": {
            "qwen": "qwen-3.8-27b",
            "deepseek_flash": "deepseek-v4-flash",
            "deepseek_pro": "deepseek-v4-pro",
            "hy3": "hy3",
        },
    },
    "ollama_cloud": {
        "base_url": "https://ollama.com/v1",
        "api_key": os.environ.get("OLLAMA_API_KEY", ""),
        "models": {"gemma": "gemma4:31b"},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "models": {
            "ox_alpha": "ox-alpha",
            "laguna_s": "laguna-s-2.1",
            "laguna_xs": "laguna-xs-2.1",
            "north_mini_code": "north-mini-code",
            "nemotron_ultra": "nemotron-3-ultra",
        },
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "models": {
            "qwen": "qwen3.6-27b",
            "gpt_oss": "gpt-oss-120b",
        },
    },
    "opencode_zen": {
        "base_url": "https://api.opencode-zen.com/v1",
        "api_key": os.environ.get("OPENCODE_ZEN_API_KEY", ""),
        "models": {
            "hy3": "hy3",
            "ox_alpha": "ox-alpha",
            "muse_spark": "muse-spark-1.2-contributor",
        },
    },
}

# Распределение ролей в агентном цикле
# Принцип: провайдеры распределены по ролям с учётом их сильных сторон
ROLE_MAPPING = {
    # Планирование/декомпозиция задач — Nemotron Ultra (сильный reasoning)
    "plan": [("openrouter", "nemotron_ultra")],
    
    # Быстрая генерация кода — Groq Qwen (LPU скорость для итеративного цикла)
    "generate": [("groq", "qwen")],
    
    # Первые попытки исправления (1-2) — скорость важна
    "fix_early": [("groq", "qwen")],
    
    # Эскалация при залипании (попытка 3+) — мощная модель Groq или DeepSeek Pro через OrcaRouter
    "fix_late": [("groq", "gpt_oss"), ("orcarouter", "deepseek_pro")],
    
    # Глубокий разбор traceback — Nemotron Ultra reasoning
    "fix_deep": [("openrouter", "nemotron_ultra")],
    
    # Critic/код-ревью — Ox Alpha или Hy3 (разная модельная семья от генератора)
    "critic": [("opencode_zen", "ox_alpha"), ("orcarouter", "hy3")],
    
    # Извлечение фактов для памяти — быстрый бесплатный вариант
    "memory": [("openrouter", "laguna_xs")],
    
    # Код-генерация с акцентом на качество — North Mini Code (бесплатно)
    "code_quality": [("openrouter", "north_mini_code")],
    
    # Резервная генерация — DeepSeek Flash (быстрый и дешёвый)
    "generate_backup": [("orcarouter", "deepseek_flash")],
}

# Fallback-цепочка при отказе провайдера (мета-шлюз)
# Groq → OrcaRouter → OpenRouter (free) → Ollama Cloud → Opencode Zen
FALLBACK_CHAIN = [
    ("groq", "qwen"),
    ("orcarouter", "qwen"),
    ("openrouter", "laguna_xs"),
    ("ollama_cloud", "gemma"),
    ("opencode_zen", "hy3"),
]
