from typing import Optional

class CustomException(Exception):
    """
    Base exception for the Medical RAG system.

    Supports:
    - error chaining
    - contextual metadata
    - readable logging
    """

    def __init__(
        self,
        message: str,
        error: Optional[Exception] = None,
        context: Optional[dict] = None,
    ):
        self.message = message
        self.original_error = error
        self.context = context or {}

        error_msg = message

        if error:
            error_msg += f" | RootError: {repr(error)}"

        if self.context:
            error_msg += f" | Context: {self.context}"

        super().__init__(error_msg)
