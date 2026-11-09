"""User-facing tool errors.

The plugin maps these to JSON-RPC errors with a stable ``data.code`` so the
bundle can show a plain-language message and choose a recovery action.
"""

from __future__ import annotations


class ToolError(Exception):
    def __init__(self, code: str, message: str, data: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "data": self.data}
