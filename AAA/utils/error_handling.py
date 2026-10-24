import traceback
from typing import Optional

def log_error(error: Exception, context: str = "") -> str:
    """Format error message with context and traceback"""
    error_message = [
        f"\n{'='*50}",
        f"ERROR in {context}:" if context else "ERROR:",
        f"{str(error)}",
        "\nTraceback:",
        traceback.format_exc(),
        f"{'='*50}\n"
    ]
    return "\n".join(error_message)
