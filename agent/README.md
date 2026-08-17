# Coding AI Agent

AI-агент для генерации и исправления кода с использованием мульти-провайдер LLM архитектуры.

## Архитектура

Агент использует 6 LLM-провайдеров с распределением по ролям:

| Провайдер | Модель | Роль |
|-----------|--------|------|
| **Groq** | qwen3.6-27B | Основной генератор кода (быстрый инференс) |
| **Groq** | gpt-oss-120B | Эскалация при сложных багах |
| **OrcaRouter** | orcarouter/auto | Планирование + встроенный failover |
| **Agnes AI** | Agnes Flash | Critic-агент (код-ревью) |
| **OpenRouter** | Nemotron | Планирование и reasoning |
| **Ollama Cloud** | Gemma4:31b | Резервный fallback |
| **Opencode Zen** | Hy3/DeepSeek | Резервный fallback + код |

## Агентный цикл

```
1. Retrieve context (Memory Layer, Qdrant)
2. Plan          → OpenRouter (Nemotron)
3. Generate      → Groq (qwen3.6-27B)
4. Execute       → sandbox (subprocess)
5. Test          → pytest
6. Fix loop:
   - попытка 1-2 → Groq (qwen3.6-27B)
   - попытка 3+  → Groq (gpt-oss-120B) или OrcaRouter
   - если падает → OpenRouter (Nemotron)
7. Critic/Review → Agnes AI
8. Store memory  → Qdrant
```

## Установка

```bash
cd agent
pip install -r requirements.txt
```

## Настройка

Установите переменные окружения для API ключей:

```bash
export GROQ_API_KEY=your_key
export ORCAROUTER_API_KEY=your_key
export AGNES_API_KEY=your_key
export OPENROUTER_API_KEY=your_key
export OLLAMA_API_KEY=your_key
export OPENCODE_ZEN_API_KEY=your_key
```

## Запуск

### API сервер

```bash
python -m agent.api.main
```

API доступно на `http://localhost:8000`

### Streamlit UI

```bash
streamlit run agent/ui/app.py
```

UI доступно на `http://localhost:8501`

## Использование через Python

```python
from agent import CodeAgent, AgentRouter, Sandbox, MemoryLayer

# Инициализация
router = AgentRouter()
sandbox = Sandbox()
memory = MemoryLayer()
memory.connect()

agent = CodeAgent(
    router=router,
    sandbox=sandbox,
    memory=memory,
    max_fix_attempts=5,
)

# Запуск задачи
result = agent.run(
    task="Напишите функцию quicksort на Python",
    context="",
    require_tests=True,
)

print(f"Success: {result['success']}")
print(f"Code: {result['code']}")
print(f"Iterations: {result['iterations']}")
```

## API Endpoints

- `GET /health` - Проверка здоровья
- `POST /run` - Запустить полный агентный цикл
- `POST /generate` - Только генерация кода
- `POST /fix` - Исправить код по ошибке

## Структура проекта

```
agent/
├── __init__.py          # Главный модуль
├── requirements.txt     # Зависимости
├── core/
│   ├── __init__.py
│   └── agent.py         # CodeAgent класс
├── providers/
│   ├── __init__.py
│   ├── config.py        # Конфигурация провайдеров
│   └── router.py        # AgentRouter класс
├── sandbox/
│   ├── __init__.py
│   └── executor.py      # Песочница для выполнения кода
├── memory/
│   ├── __init__.py
│   └── layer.py         # MemoryLayer (Qdrant)
├── api/
│   ├── __init__.py
│   └── main.py          # FastAPI приложение
└── ui/
    ├── __init__.py
    └── app.py           # Streamlit UI
```

## Особенности

- **Многоуровневый fallback**: При отказе провайдера автоматически переключается на следующий
- **Разделение ролей**: Critic использует другую модельную семью чем Generator (снижение correlated failure)
- **Эскалация**: При repeated failures переключается на более мощные модели
- **Память**: Сохранение успешных решений в Qdrant для будущего использования
