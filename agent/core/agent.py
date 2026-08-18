"""Основной модуль агента для генерации и исправления кода."""

from typing import Optional
import structlog

from ..providers.router import AgentRouter
from ..providers.config import ROLE_MAPPING, FALLBACK_CHAIN
from ..sandbox.executor import Sandbox
from ..memory.layer import MemoryLayer

logger = structlog.get_logger(__name__)


class CodeAgent:
    """
    AI-агент для генерации и исправления кода.
    
    Реализует агентный цикл:
    1. Retrieve context (Memory Layer)
    2. Plan (OpenRouter Nemotron)
    3. Generate (Groq qwen3.6-27B)
    4. Execute (sandbox)
    5. Test (pytest)
    6. Fix loop (с эскалацией)
    7. Critic/Review (Agnes Flash)
    8. Store memory
    """

    def __init__(
        self,
        router: Optional[AgentRouter] = None,
        sandbox: Optional[Sandbox] = None,
        memory: Optional[MemoryLayer] = None,
        max_fix_attempts: int = 5,
    ):
        self.router = router or AgentRouter()
        self.sandbox = sandbox or Sandbox()
        self.memory = memory
        self.max_fix_attempts = max_fix_attempts
        logger.info("CodeAgent initialized")

    def plan(self, task: str, context: str = "") -> str:
        """
        Шаг планирования: декомпозиция задачи.
        
        Args:
            task: Описание задачи
            context: Дополнительный контекст
            
        Returns:
            План выполнения задачи
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a planning assistant. "
                    "Break down the given task into clear, actionable steps. "
                    "Output a numbered list of steps."
                )
            },
            {
                "role": "user",
                "content": f"Task: {task}\n\nContext: {context}" if context else f"Task: {task}",
            }
        ]
        
        fallback = ROLE_MAPPING.get("plan", [("openrouter", "planner")])
        response = self.router.call(role="plan", messages=messages, fallback_chain=fallback)
        
        logger.info(f"Plan generated: {response[:100]}...")
        return response

    def generate(self, plan: str, context: str = "") -> str:
        """
        Шаг генерации: создание кода по плану.
        
        Args:
            plan: План выполнения
            context: Дополнительный контекст
            
        Returns:
            Сгенерированный код
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer. "
                    "Write clean, efficient, and well-documented code. "
                    "Include error handling and follow best practices."
                )
            },
            {
                "role": "user",
                "content": f"Implement this plan:\n{plan}\n\nContext: {context}" if context else f"Implement this plan:\n{plan}",
            }
        ]
        
        fallback = ROLE_MAPPING.get("generate", [("groq", "fast")])
        response = self.router.call(
            role="generate",
            messages=messages,
            fallback_chain=fallback,
            temperature=0.3,
        )
        
        logger.info(f"Code generated ({len(response)} chars)")
        return response

    def execute_and_test(self, code: str, test_code: str = "") -> tuple[bool, str]:
        """
        Выполнить код и запустить тесты.
        
        Args:
            code: Код для выполнения
            test_code: Тестовый код (опционально)
            
        Returns:
            (success, output/error message)
        """
        # Выполнение основного кода
        returncode, stdout, stderr = self.sandbox.execute(code)
        
        if returncode != 0:
            logger.warning(f"Code execution failed: {stderr}")
            return False, stderr
        
        # Запуск тестов если предоставлены
        if test_code:
            success, test_stdout, test_stderr = self.sandbox.run_tests(test_code)
            if not success:
                logger.warning(f"Tests failed: {test_stderr}")
                return False, test_stderr
        
        logger.info("Code executed successfully")
        return True, stdout

    def fix(
        self,
        code: str,
        error: str,
        attempt: int,
    ) -> str:
        """
        Исправить код на основе ошибки.
        
        Args:
            code: Исходный код
            error: Сообщение об ошибке
            attempt: Номер попытки (для эскалации)
            
        Returns:
            Исправленный код
        """
        # Выбор стратегии в зависимости от номера попытки
        if attempt <= 2:
            fallback = ROLE_MAPPING.get("fix_early", [("groq", "fast")])
        elif attempt <= 4:
            fallback = ROLE_MAPPING.get("fix_late", [("groq", "strong"), ("orcarouter", "auto")])
        else:
            fallback = ROLE_MAPPING.get("fix_deep", [("openrouter", "planner")])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a debugging expert. "
                    "Analyze the error and fix the code. "
                    "Explain what was wrong and how you fixed it."
                )
            },
            {
                "role": "user",
                "content": f"Original code:\n{code}\n\nError:\n{error}\n\nFix the code.",
            }
        ]
        
        response = self.router.call(
            role="fix",
            messages=messages,
            fallback_chain=fallback,
            temperature=0.3,
        )
        
        logger.info(f"Fix attempt {attempt} completed")
        return response

    def critic_review(self, code: str, task: str) -> dict:
        """
        Код-ревью сгенерированного решения.
        
        Args:
            code: Код для ревью
            task: Исходная задача
            
        Returns:
            Словарь с результатами ревью
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code reviewer. "
                    "Evaluate the code for correctness, efficiency, and best practices. "
                    "Provide constructive feedback."
                )
            },
            {
                "role": "user",
                "content": f"Task: {task}\n\nCode to review:\n{code}",
            }
        ]
        
        fallback = ROLE_MAPPING.get("critic", [("agnes", "critic")])
        response = self.router.call(
            role="critic",
            messages=messages,
            fallback_chain=fallback,
        )
        
        # Парсинг ответа (упрощённо)
        review = {
            "approved": "APPROVED" in response.upper() or "LGTM" in response.upper(),
            "feedback": response,
        }
        
        logger.info(f"Critic review completed, approved={review['approved']}")
        return review

    def run(
        self,
        task: str,
        context: str = "",
        require_tests: bool = True,
    ) -> dict:
        """
        Запустить полный агентный цикл.
        
        Args:
            task: Описание задачи
            context: Дополнительный контекст
            require_tests: Требовать ли тесты
            
        Returns:
            Результат выполнения: {success, code, output, iterations}
        """
        result = {
            "success": False,
            "code": "",
            "output": "",
            "iterations": 0,
            "error": None,
        }
        
        try:
            # Шаг 1: Retrieve context from memory
            if self.memory:
                memory_context = self.memory.retrieve_context(task)
                if memory_context:
                    context = f"{context}\n\nRetrieved from memory:\n{memory_context}"
            
            # Шаг 2: Plan
            plan = self.plan(task, context)
            logger.info(f"Plan: {plan[:200]}...")
            
            # Шаг 3: Generate
            code = self.generate(plan, context)
            result["code"] = code
            
            # Генерация тестов если требуется
            test_code = ""
            if require_tests:
                test_messages = [
                    {
                        "role": "system",
                        "content": "Generate pytest tests for the given code.",
                    },
                    {
                        "role": "user",
                        "content": f"Code:\n{code}\n\nGenerate comprehensive tests.",
                    }
                ]
                test_code = self.router.call(
                    role="generate",
                    messages=test_messages,
                    fallback_chain=[("groq", "fast")],
                )
            
            # Шаг 4-6: Execute/Test/Fix loop
            for attempt in range(1, self.max_fix_attempts + 1):
                result["iterations"] = attempt
                
                success, output = self.execute_and_test(code, test_code)
                
                if success:
                    # Шаг 7: Critic review
                    review = self.critic_review(code, task)
                    
                    if review["approved"]:
                        result["success"] = True
                        result["output"] = output
                        
                        # Шаг 8: Store memory
                        if self.memory:
                            self.memory.store_fact(
                                f"Task: {task}\nSolution: {code[:500]}",
                                metadata={"status": "success"},
                            )
                        
                        logger.info(f"Task completed successfully in {attempt} iterations")
                        return result
                    else:
                        # Критик не одобрил, нужно исправить
                        error_msg = f"Critic feedback:\n{review['feedback']}"
                else:
                    error_msg = output
                
                # Попытка исправления
                if attempt < self.max_fix_attempts:
                    code = self.fix(code, error_msg, attempt)
                    result["code"] = code
                else:
                    result["error"] = error_msg
                    logger.warning(f"Max attempts reached, last error: {error_msg}")
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            logger.exception(f"Agent run failed: {e}")
            return result
