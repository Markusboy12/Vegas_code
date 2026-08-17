# Финальное распределение провайдеров по ролям в агенте

## Принцип распределения

Шесть провайдеров — избыточно как "просто список альтернатив", но осмысленно как **распределение по ролям в агентном цикле**, где у каждого своя сильная сторона (скорость / контекст / отказоустойчивость). OrcaRouter при этом выполняет функцию мета-шлюза с собственным failover, а остальные пять — это независимый резервный слой на случай, если сам OrcaRouter или конкретная модель внутри него недоступны.

## Таблица ролей

| Провайдер | Модель | Роль в агенте | Почему |
|---|---|---|---|
| **Groq** | qwen3.6-27B | Основной генератор кода (Generate/Fix loop, ранние попытки) | Самый быстрый инференс (LPU) — критично для итеративного цикла с частыми короткими вызовами |
| **Groq** | gpt-oss-120B | Эскалация при "залипании" (после 2-3 неудачных попыток fix loop) | Крупнее по параметрам, лучше держит сложные баги, но с бóльшим "весом" запроса |
| **OrcaRouter** | Qwen-3.8-27B / `orcarouter/auto` | Основной шлюз для планирования и general-purpose шагов, плюс встроенный fallback по 200+ моделям | Собственная адаптивная маршрутизация и failover избавляют от части ручной логики |
| **Agnes AI** | Agnes Flash 2.0/2.5 | Critic-агент (код-ревью) и работа с большим контекстом (чтение всего репозитория) | Другая модельная семья, чем генератор (снижает shared bias генератор↔критик), контекст 256K-512K хорошо ложится на "прочитать весь проект целиком" |
| **OpenRouter** | Nemotron-серия | Планирование/декомпозиция задач, разбор сложных traceback | Nemotron славится сильным reasoning — хорош там, где нужно именно рассуждение, а не быстрая генерация кода |
| **Ollama Cloud** | Gemma4:31b | Резервный пул №1 (если Groq/OrcaRouter недоступны) | Бесплатно, без карты, но без гарантии SLA — только как fallback, не основной путь |
| **Opencode Zen** | Hy3 / DeepSeek | Резервный пул №2 (последний fallback) + DeepSeek как альтернативный вариант для генерации кода | DeepSeek исторически силён именно в код-специфичных задачах — можно тестировать как альтернативу основному генератору |

## Обновлённая схема агентного цикла с ролями

```
1. Retrieve context (Memory Layer, Qdrant)
2. Plan          → OpenRouter (Nemotron) — декомпозиция задачи
3. Generate       → Groq (qwen3.6-27B) — быстрая генерация кода
4. Execute        → sandbox (subprocess/Judge0)
5. Test           → pytest
6. Fix loop:
   попытка 1-2    → Groq (qwen3.6-27B)
   попытка 3+     → Groq (gpt-oss-120B) или OrcaRouter auto
   если всё ещё падает → OpenRouter (Nemotron) — глубокий разбор traceback
7. Critic/Review  → Agnes AI (Agnes Flash) — ревью перед принятием
8. Store memory   → любой доступный провайдер для извлечения фактов (не критично к скорости)

Fallback-цепочка при отказе провайдера:
Groq → OrcaRouter → Ollama Cloud → Opencode Zen
```

## Router-класс — реализация с приоритетом и ролями

```python
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import os

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ["GROQ_API_KEY"],
        "models": {"fast": "qwen3.6-27b", "strong": "gpt-oss-120b"},
    },
    "orcarouter": {
        "base_url": "https://api.orcarouter.ai/v1",
        "api_key": os.environ["ORCAROUTER_API_KEY"],
        "models": {"auto": "orcarouter/auto", "qwen": "qwen-3.8-27b"},
    },
    "agnes": {
        "base_url": "https://api.agnes-ai.com/v1",  # уточнить актуальный base_url в доке
        "api_key": os.environ["AGNES_API_KEY"],
        "models": {"critic": "agnes-2.5-flash"},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "models": {"planner": "nvidia/nemotron-..."},  # уточнить точный slug модели
    },
    "ollama_cloud": {
        "base_url": "https://ollama.com/v1",  # уточнить актуальный base_url
        "api_key": os.environ["OLLAMA_API_KEY"],
        "models": {"fallback": "gemma4:31b"},
    },
    "opencode_zen": {
        "base_url": "...",  # уточнить у провайдера
        "api_key": os.environ["OPENCODE_ZEN_API_KEY"],
        "models": {"fallback": "hy3", "deepseek": "deepseek-..."},
    },
}

class AgentRouter:
    def __init__(self):
        self.clients = {
            name: OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
            for name, cfg in PROVIDERS.items()
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call(self, role: str, messages: list, fallback_chain: list[tuple[str, str]]):
        """
        role: логическая роль (generate/plan/critic/fix)
        fallback_chain: [(provider_name, model_key), ...] в порядке приоритета
        """
        last_error = None
        for provider_name, model_key in fallback_chain:
            try:
                client = self.clients[provider_name]
                model = PROVIDERS[provider_name]["models"][model_key]
                return client.chat.completions.create(model=model, messages=messages)
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"Все провайдеры недоступны для роли {role}: {last_error}")


router = AgentRouter()

# Пример вызова для fix loop с эскалацией
response = router.call(
    role="fix",
    messages=[...],
    fallback_chain=[
        ("groq", "fast"),
        ("orcarouter", "auto"),
        ("ollama_cloud", "fallback"),
        ("opencode_zen", "fallback"),
    ],
)
```

⚠️ Base URL и точные model slug для OpenRouter (Nemotron), Ollama Cloud и Opencode Zen нужно свериться с актуальной документацией каждого провайдера перед стартом — они могут отличаться от указанных здесь плейсхолдеров.

## Обновлённый requirements.txt

```txt
fastapi==0.116.0
uvicorn[standard]==0.35.0
pydantic==2.11.0
openai==1.99.0          # единый клиент для ВСЕХ провайдеров (все OpenAI-совместимы)
tenacity==9.1.0
httpx==0.28.1
qdrant-client==1.15.0
GitPython==3.1.45
pytest==8.4.0
pytest-timeout==2.4.0
structlog==25.1.0
streamlit==1.48.0
```

Все шесть провайдеров OpenAI-совместимы, поэтому по-прежнему достаточно одного SDK (`openai`) с разными `base_url` — никаких конфликтов зависимостей от специфичных клиентских библиотек.

## Что подчеркнуть на собеседовании

- Осознанное распределение ролей между провайдерами (не "случайный набор бесплатных ключей", а решение под конкретные сильные стороны: скорость Groq, контекст Agnes, reasoning Nemotron, отказоустойчивость OrcaRouter)
- Многоуровневый fallback: сначала логика внутри OrcaRouter, затем — свой ручной fallback между независимыми провайдерами на уровень выше (defense in depth против отказа целого шлюза, а не только одной модели)
- Почему Critic намеренно использует другую модельную семью (Agnes), чем Generator (Qwen на Groq) — снижение correlated failure между генерацией и проверкой
