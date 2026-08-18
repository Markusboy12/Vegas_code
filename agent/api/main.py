"""FastAPI приложение для Coding AI Agent."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import structlog

from agent import CodeAgent, AgentRouter, Sandbox, MemoryLayer

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Coding AI Agent API",
    description="API для генерации и исправления кода с использованием мульти-провайдер LLM архитектуры",
    version="1.0.0",
)

# Глобальные компоненты
router: Optional[AgentRouter] = None
sandbox: Optional[Sandbox] = None
memory: Optional[MemoryLayer] = None
agent: Optional[CodeAgent] = None


class TaskRequest(BaseModel):
    """Запрос на выполнение задачи."""
    task: str
    context: Optional[str] = ""
    require_tests: bool = True
    max_iterations: int = 5


class TaskResponse(BaseModel):
    """Ответ на задачу."""
    success: bool
    code: str
    output: str
    iterations: int
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Ответ проверки здоровья."""
    status: str
    providers_configured: int


@app.on_event("startup")
async def startup_event():
    """Инициализация компонентов при старте."""
    global router, sandbox, memory, agent
    
    router = AgentRouter()
    sandbox = Sandbox()
    memory = MemoryLayer()
    
    # Попытка подключения к Qdrant (не критично)
    memory.connect()
    
    agent = CodeAgent(
        router=router,
        sandbox=sandbox,
        memory=memory,
        max_fix_attempts=5,
    )
    
    logger.info("Agent initialized")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса."""
    providers_count = len(router.clients) if router else 0
    
    return HealthResponse(
        status="healthy" if providers_count > 0 else "degraded",
        providers_configured=providers_count,
    )


@app.post("/run", response_model=TaskResponse)
async def run_task(request: TaskRequest):
    """
    Выполнить задачу по генерации/исправлению кода.
    
    Запускает полный агентный цикл:
    1. Планирование
    2. Генерация кода
    3. Выполнение и тестирование
    4. Исправление ошибок (fix loop)
    5. Код-ревью
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = agent.run(
            task=request.task,
            context=request.context or "",
            require_tests=request.require_tests,
        )
        
        logger.info(
            f"Task completed",
            success=result["success"],
            iterations=result["iterations"]
        )
        
        return TaskResponse(**result)
        
    except Exception as e:
        logger.exception(f"Task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_code(task: str, context: Optional[str] = ""):
    """Только генерация кода без выполнения."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    plan = agent.plan(task, context or "")
    code = agent.generate(plan, context or "")
    
    return {"plan": plan, "code": code}


@app.post("/fix")
async def fix_code(code: str, error: str, attempt: int = 1):
    """Исправить код на основе ошибки."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    fixed_code = agent.fix(code, error, attempt)
    return {"fixed_code": fixed_code}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
