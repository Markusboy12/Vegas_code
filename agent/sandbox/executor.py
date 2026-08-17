"""Модуль песочницы для безопасного выполнения кода."""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class Sandbox:
    """
    Песочница для безопасного выполнения сгенерированного кода.
    
    Поддерживает:
    - Выполнение Python-кода через subprocess
    - Запуск pytest для тестирования
    - Ограничение по времени выполнения
    """

    def __init__(self, workdir: Optional[str] = None):
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp())
        self.workdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Sandbox initialized at {self.workdir}")

    def execute(
        self,
        code: str,
        timeout: int = 30,
        args: Optional[list[str]] = None,
    ) -> tuple[int, str, str]:
        """
        Выполнить Python-код в песочнице.
        
        Args:
            code: Исходный код для выполнения
            timeout: Таймаут выполнения в секундах
            args: Аргументы командной строки
            
        Returns:
            (return_code, stdout, stderr)
        """
        script_path = self.workdir / "script.py"
        script_path.write_text(code)
        
        cmd = ["python", str(script_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"Executing code in sandbox, timeout={timeout}s")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workdir),
            )
            logger.info(
                f"Execution completed",
                returncode=result.returncode,
                stdout_len=len(result.stdout),
                stderr_len=len(result.stderr)
            )
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Execution timed out after {timeout}s")
            return -1, "", f"Timeout after {timeout} seconds"
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return -1, "", str(e)

    def run_tests(
        self,
        test_code: str,
        timeout: int = 60,
    ) -> tuple[bool, str, str]:
        """
        Запустить pytest на тестовом коде.
        
        Args:
            test_code: Код тестов
            timeout: Таймаут выполнения
            
        Returns:
            (success, stdout, stderr)
        """
        test_path = self.workdir / "test_script.py"
        test_path.write_text(test_code)
        
        cmd = [
            "python", "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            f"--timeout={timeout}",
        ]
        
        logger.info("Running pytest in sandbox")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
                cwd=str(self.workdir),
            )
            success = result.returncode == 0
            logger.info(
                f"Tests completed",
                success=success,
                returncode=result.returncode
            )
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Tests timed out after {timeout}s")
            return False, "", f"Tests timeout after {timeout} seconds"
        except Exception as e:
            logger.error(f"Tests failed: {e}")
            return False, "", str(e)

    def cleanup(self):
        """Очистить временные файлы."""
        import shutil
        try:
            shutil.rmtree(self.workdir)
            logger.info(f"Sandbox cleaned up: {self.workdir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup sandbox: {e}")
