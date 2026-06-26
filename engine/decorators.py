"""Logging and timing decorators for the ingestion pipeline steps."""

import functools
import time
from typing import Any, Callable
from core.logging import setup_logger

logger = setup_logger("engine.steps", "preprocessing/pipeline.log")


def log_pipeline_step(step_name: str) -> Callable:
    """Decorator that wraps pipeline steps, logs execution timing, inputs, and exceptions."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info(f"Pipeline Step: Started '{step_name}'")
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                logger.info(f"Pipeline Step: Finished '{step_name}' in {duration:.4f}s")
                
                # Check if history list is passed directly in parameters or exists as an attribute on self
                history_list = kwargs.get("history")
                if history_list is None:
                    for arg in args[1:]:
                        if isinstance(arg, list):
                            history_list = arg
                            break
                
                if history_list is not None:
                    history_list.append(f"{step_name} completed in {duration:.4f}s")
                elif args and hasattr(args[0], "history") and isinstance(args[0].history, list):
                    args[0].history.append(f"{step_name} completed in {duration:.4f}s")
                
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(
                    f"Pipeline Step: Failed '{step_name}' after {duration:.4f}s. Error: {str(e)}"
                )
                raise e

        return wrapper

    return decorator
