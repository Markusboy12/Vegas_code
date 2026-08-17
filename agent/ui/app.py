"""Streamlit UI для Coding AI Agent."""

import streamlit as st
from agent import CodeAgent, AgentRouter, Sandbox, MemoryLayer

st.set_page_config(
    page_title="Coding AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Coding AI Agent")
st.markdown("""
AI-агент для генерации и исправления кода с использованием мульти-провайдер LLM архитектуры.

**Провайдеры:**
- Groq (быстрая генерация)
- OrcaRouter (адаптивная маршрутизация)
- Agnes AI (код-ревью)
- OpenRouter (планирование)
- Ollama Cloud (fallback)
- Opencode Zen (fallback)
""")

# Инициализация агента
@st.cache_resource
def get_agent():
    router = AgentRouter()
    sandbox = Sandbox()
    memory = MemoryLayer()
    memory.connect()
    return CodeAgent(router=router, sandbox=sandbox, memory=memory)

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Failed to initialize agent: {e}")
    agent = None

# Боковая панель
with st.sidebar:
    st.header("Настройки")
    
    max_iterations = st.slider(
        "Максимум итераций",
        min_value=1,
        max_value=10,
        value=5,
    )
    
    require_tests = st.checkbox("Генерировать тесты", value=True)
    
    if st.button("Проверить здоровье"):
        if agent:
            st.success("Агент инициализирован")
            st.write(f"Провайдеров настроено: {len(agent.router.clients)}")
        else:
            st.error("Агент не инициализирован")

# Основная форма
task = st.text_area(
    "Описание задачи",
    placeholder="Опишите задачу, которую нужно решить...",
    height=150,
)

context = st.text_area(
    "Дополнительный контекст (опционально)",
    placeholder="Любая дополнительная информация...",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("Запустить", type="primary", disabled=not agent)

if run_button and task and agent:
    with st.spinner("Выполнение задачи..."):
        try:
            result = agent.run(
                task=task,
                context=context,
                require_tests=require_tests,
            )
            
            # Отображение результатов
            if result["success"]:
                st.success(f"✅ Задача выполнена за {result['iterations']} итераций")
            else:
                st.warning(f"⚠️ Задача не выполнена: {result.get('error', 'Неизвестная ошибка')}")
            
            # План
            if "plan" in result:
                with st.expander("План выполнения"):
                    st.markdown(result["plan"])
            
            # Код
            with st.expander("Сгенерированный код", expanded=True):
                st.code(result["code"], language="python")
            
            # Вывод
            if result["output"]:
                with st.expander("Вывод программы"):
                    st.text(result["output"])
            
            # Статистика
            st.metric("Итераций", result["iterations"])
            
        except Exception as e:
            st.error(f"Ошибка: {e}")

elif run_button and not task:
    st.warning("Введите описание задачи")

elif not agent:
    st.info("Агент не инициализирован. Проверьте настройки API ключей.")

# Примеры задач
with st.expander("Примеры задач"):
    st.markdown("""
    1. **Написать функцию сортировки**: 
       "Напишите функцию quicksort на Python для сортировки списка чисел"
    
    2. **Исправить баг**: 
       "В функции есть ошибка: она возвращает None вместо пустого списка"
    
    3. **Создать класс**: 
       "Создайте класс BankAccount с методами deposit, withdraw и get_balance"
    """)
