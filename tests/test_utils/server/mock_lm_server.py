"""Lightweight stdlib HTTP server for integration tests."""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

TEST_SERVER_LOG_FILE_PATH_ENV_VAR = "TEST_SERVER_LOG_FILE_PATH"


def _read_body(handler):
    length = int(handler.headers.get("Content-Length", 0))
    return json.loads(handler.rfile.read(length)) if length else {}


def _send_json(handler, data, status=200):
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(handler, status, message, error_type="error"):
    _send_json(handler, {"error": {"message": message, "type": error_type, "code": status}}, status=status)


def _check_error_injection(content):
    """Return (status, message) if content triggers error injection, else None."""
    if "429" in content:
        return 429, "Rate limit exceeded"
    if "504" in content:
        return 504, "Request timed out"
    if "400" in content:
        return 400, "Bad request"
    if "401" in content:
        return 401, "Authentication error"
    return None


def _extract_content(body):
    """Extract text content from request messages for error injection checks."""
    messages = body.get("messages", [])
    if messages:
        msg = messages[0]
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle content blocks format
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
            return ""
        return str(content)
    # For responses API, check input
    inp = body.get("input", "")
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    return content
        return ""
    return str(inp)


def _log_request(body):
    log_path = os.environ.get(TEST_SERVER_LOG_FILE_PATH_ENV_VAR)
    if not log_path:
        return
    with open(log_path, "a") as f:
        log_blob = ({"model": body.get("model", ""), "messages": body.get("messages", [])},)
        json.dump(log_blob, f)
        f.write("\n")


def _chat_completion_response(body):
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def _chat_completion_stream_response(body):
    """Return a list of SSE chunks for streaming chat completions."""
    chunks = [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "gpt-4o"),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": '{"output_text": "Hello!"}'},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "gpt-4o"),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        },
    ]
    return chunks


def _text_completion_response(body):
    return {
        "id": "cmpl-mock",
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o"),
        "choices": [
            {
                "index": 0,
                "text": "Hi!",
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def _responses_api_response(body):
    return {
        "id": "resp-mock",
        "object": "response",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o"),
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi!"}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    }


class MockLMHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible API requests with mock responses."""

    def log_message(self, format, *args):
        # Suppress default stderr logging
        pass

    def do_POST(self):
        body = _read_body(self)
        content = _extract_content(body)

        # Error injection
        error = _check_error_injection(content)
        if error:
            _send_error(self, error[0], error[1])
            return

        _log_request(body)

        # Strip query string for matching, normalize path
        path = self.path.split("?")[0].rstrip("/")

        # Match endpoint by suffix to handle /v1/..., /chat/..., and
        # Azure-style /openai/deployments/{model}/... paths
        if path.endswith("/chat/completions"):
            if body.get("stream"):
                self._send_stream(_chat_completion_stream_response(body))
            else:
                _send_json(self, _chat_completion_response(body))
        elif path.endswith("/completions"):
            _send_json(self, _text_completion_response(body))
        elif path.endswith("/responses"):
            _send_json(self, _responses_api_response(body))
        else:
            _send_error(self, 404, f"Not found: {self.path}")

    def _send_stream(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            line = f"data: {json.dumps(chunk)}\n\n"
            self.wfile.write(line.encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        # Health check
        if self.path in ("/health", "/v1/health"):
            _send_json(self, {"status": "ok"})
        else:
            _send_error(self, 404, f"Not found: {self.path}")


def start_server(port, host="127.0.0.1"):
    """Create and return an HTTPServer instance (not yet serving)."""
    server = HTTPServer((host, port), MockLMHandler)
    return server
