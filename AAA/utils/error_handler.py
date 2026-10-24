import logging
import functools
import traceback
from typing import Callable, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ikaggle')

def log_error(e: Exception, msg: str = "") -> str:
    """Format error message with full traceback"""
    tb = traceback.extract_tb(e.__traceback__)
    error_msg = f"\n{msg} Error:\n"
    error_msg += f"File: {tb[-1].filename}, line {tb[-1].lineno}\n"
    error_msg += f"Function: {tb[-1].name}\n"
    error_msg += f"Code: {tb[-1].line}\n"
    error_msg += f"Exception: {type(e).__name__}: {str(e)}\n"
    error_msg += f"Full traceback:\n{traceback.format_exc()}"
    return error_msg

def with_error_handling(func: Callable) -> Callable:
    """Decorator to handle errors with proper logging"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = log_error(e, f"Error in {func.__name__}")
            logger.error(error_msg)
            raise type(e)(error_msg) from e
    return wrapper
