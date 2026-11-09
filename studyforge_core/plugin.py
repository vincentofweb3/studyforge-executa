"""StudyForge Executa plugin: JSON-RPC over stdio, long-running, v2 sampling."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .errors import ToolError
from .sampling import SamplingClient, SamplingError
from .tools import ACTIONS, dispatch

MANIFEST: dict[str, Any] = {
    "display_name": "StudyForge Pack Builder",
    "version": "0.1.0",
    "description": (
        "Builds grounded study packs from notes or a topic: concepts, flashcards, "
        "quiz items, a day-by-day plan, mastery and review scheduling."
    ),
    "author": "StudyForge",
    "host_capabilities": ["llm.sample"],
    "tools": [
        {
            "name": "ping",
            "description": "Smoke-test method for the StudyForge tool.",
            "parameters": [],
        },
        {
            "name": "studyforge",
            "description": (
                "StudyForge dispatcher. Actions: " + ", ".join(ACTIONS) + "."
            ),
            "parameters": [
                {
                    "name": "action",
                    "type": "string",
                    "description": "One of the StudyForge actions.",
                    "required": True,
                },
                {
                    "name": "payload",
                    "type": "object",
                    "description": "Action-specific arguments.",
                    "required": False,
                    "default": {},
                },
            ],
        },
    ],
    "runtime": {"type": "uv", "min_version": "0.1.0"},
}

_STDOUT_LOCK = threading.Lock()


def _write_frame(message: dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False)
    with _STDOUT_LOCK:
        sys.stdout.write(payload + "\n")
        sys.stdout.flush()


SAMPLING = SamplingClient(write_frame=_write_frame)


def _response(req_id: Any, *, result: Any = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    return out


def _handle_initialize(req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    offered = str(params.get("protocolVersion") or "1.1")
    if offered != "2.0":
        SAMPLING.disable(
            f"host did not negotiate v2 (offered protocolVersion={offered!r}); "
            "sampling/createMessage requires Executa protocol 2.0"
        )
    sampling_caps = {"sampling": {}} if SAMPLING.enabled else {}
    return _response(
        req_id,
        result={
            "protocolVersion": offered if offered in ("1.1", "2.0") else "2.0",
            "server_info": {
                "name": MANIFEST["display_name"],
                "version": MANIFEST["version"],
            },
            "serverInfo": {
                "name": MANIFEST["display_name"],
                "version": MANIFEST["version"],
            },
            "capabilities": sampling_caps,
            "client_capabilities": sampling_caps,
        },
    )


def _handle_invoke(req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    tool = str(params.get("tool") or params.get("method") or "")
    arguments = params.get("arguments") or params.get("args") or {}
    context = params.get("context") or {}
    invoke_id = context.get("invoke_id") or params.get("invoke_id")
    try:
        data = dispatch(tool, arguments if isinstance(arguments, dict) else {}, SAMPLING, invoke_id)
        return _response(req_id, result={"success": True, "tool": tool, "data": data})
    except ToolError as exc:
        return _response(
            req_id,
            error={
                "code": -32603,
                "message": exc.message,
                "data": {"code": exc.code, **exc.data},
            },
        )
    except SamplingError as exc:
        return _response(
            req_id,
            error={"code": exc.code, "message": exc.message, "data": exc.data},
        )
    except Exception as exc:  # noqa: BLE001
        return _response(
            req_id,
            error={"code": -32603, "message": f"StudyForge failed: {exc}"},
        )


def _handle_message(message: dict[str, Any]) -> None:
    method = message.get("method")
    req_id = message.get("id")
    params = message.get("params") or {}
    if method == "initialize":
        response = _handle_initialize(req_id, params)
    elif method == "describe":
        response = _response(req_id, result=MANIFEST)
    elif method == "health":
        response = _response(
            req_id,
            result={
                "status": "ready",
                "version": MANIFEST["version"],
                "sampling": {
                    "enabled": SAMPLING.enabled,
                    "reason": SAMPLING.disabled_reason,
                },
                "actions": list(ACTIONS),
            },
        )
    elif method == "invoke":
        response = _handle_invoke(req_id, params)
    elif method == "shutdown":
        response = _response(req_id, result={"ok": True})
    else:
        response = _response(
            req_id,
            error={"code": -32601, "message": f"Method not found: {method}"},
        )
    if req_id is not None:
        _write_frame(response)


def main() -> None:
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="studyforge-invoke")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                _write_frame(_response(None, error={"code": -32700, "message": "Parse error"}))
                continue
            if not isinstance(message, dict):
                continue
            if "method" not in message:
                if not SAMPLING.dispatch_response(message):
                    print(f"unmatched sampling response id={message.get('id')!r}", file=sys.stderr)
                continue
            pool.submit(_handle_message, message)
    finally:
        pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
