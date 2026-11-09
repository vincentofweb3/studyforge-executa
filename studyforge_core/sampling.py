"""Minimal thread-safe host sampling client.

The plugin runs each invoke in a worker thread. A reader thread owns stdin and
calls :meth:`SamplingClient.dispatch_response` for frames that have no
``method`` field; worker threads block on an Event until their response arrives.
Every request carries ``context.invoke_id`` so concurrent invokes stay
correlated (see the platform's invoke-context rule).
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Callable

MAX_SAMPLING_TOKENS = 8_192


class SamplingError(Exception):
    def __init__(self, code: int, message: str, data: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "data": self.data}


class SamplingClient:
    def __init__(self, write_frame: Callable[[dict[str, Any]], None]) -> None:
        self._write_frame = write_frame
        self._pending: dict[str, tuple[threading.Event, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._disabled_reason: str | None = None

    def disable(self, reason: str) -> None:
        self._disabled_reason = reason

    @property
    def enabled(self) -> bool:
        return self._disabled_reason is None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def is_response_envelope(self, msg: dict[str, Any]) -> bool:
        if not isinstance(msg, dict) or "method" in msg:
            return False
        req_id = msg.get("id")
        with self._lock:
            return req_id is not None and req_id in self._pending

    def dispatch_response(self, msg: dict[str, Any]) -> bool:
        if not isinstance(msg, dict) or "method" in msg:
            return False
        req_id = msg.get("id")
        if req_id is None:
            return False
        with self._lock:
            pending = self._pending.pop(req_id, None)
        if pending is None:
            return False
        event, holder = pending
        holder["msg"] = msg
        event.set()
        return True

    def create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float | None = 0.2,
        response_format: dict[str, Any] | None = None,
        on_unsupported: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float = 60.0,
        invoke_id: str | None = None,
    ) -> dict[str, Any]:
        if self._disabled_reason:
            raise SamplingError(-32008, self._disabled_reason)
        if not messages:
            raise SamplingError(-32004, "sampling/createMessage needs messages")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise SamplingError(-32004, "max_tokens must be a positive integer")
        max_tokens = min(max_tokens, MAX_SAMPLING_TOKENS)

        req_id = uuid.uuid4().hex
        params: dict[str, Any] = {
            "messages": messages,
            "maxTokens": max_tokens,
            "includeContext": "none",
            "_clientTimeoutS": float(timeout),
        }
        if system_prompt is not None:
            params["systemPrompt"] = system_prompt
        if temperature is not None:
            params["temperature"] = temperature
        if response_format is not None:
            params["responseFormat"] = response_format
        if on_unsupported is not None:
            params["onUnsupported"] = on_unsupported
        if metadata:
            params["metadata"] = metadata
        if invoke_id:
            params["context"] = {"invoke_id": invoke_id}

        event = threading.Event()
        holder: dict[str, Any] = {}
        with self._lock:
            self._pending[req_id] = (event, holder)
        try:
            self._write_frame(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "sampling/createMessage",
                    "params": params,
                }
            )
        except Exception:
            with self._lock:
                self._pending.pop(req_id, None)
            raise
        if not event.wait(timeout=timeout):
            with self._lock:
                self._pending.pop(req_id, None)
            raise SamplingError(-32005, f"sampling/createMessage timed out after {timeout}s")
        response = holder.get("msg") or {}
        error = response.get("error")
        if error:
            raise SamplingError(
                int(error.get("code", -32603)),
                str(error.get("message", "sampling failed")),
                error.get("data"),
            )
        return response.get("result") or {}
