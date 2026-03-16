import json
import os
import socket
import tempfile
import threading
from typing import Any

import pytest

from tests.test_utils.server.mock_lm_server import TEST_SERVER_LOG_FILE_PATH_ENV_VAR, start_server


@pytest.fixture()
def mock_lm_test_server() -> tuple[str, str]:
    """
    Start a mock LM test server for DSPy integration tests, and tear it down
    when the test case completes.
    """
    with tempfile.TemporaryDirectory() as server_log_dir_path:
        server_log_file_path = os.path.join(server_log_dir_path, "request_logs.jsonl")
        open(server_log_file_path, "a").close()

        port = _get_random_port()
        host = "127.0.0.1"

        os.environ[TEST_SERVER_LOG_FILE_PATH_ENV_VAR] = server_log_file_path

        server = start_server(port, host)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        server_url = f"http://{host}:{port}"
        yield server_url, server_log_file_path

        server.shutdown()
        thread.join(timeout=5)
        os.environ.pop(TEST_SERVER_LOG_FILE_PATH_ENV_VAR, None)


def read_mock_lm_test_server_request_logs(server_log_file_path: str) -> list[dict[str, Any]]:
    """
    Read request logs from the mock LM server used during DSPy integration tests.

    Args:
        server_log_file_path: The filesystem path to the server request logs jsonlines file.
    Return:
        A list of log entries, where each entry corresponds to one request handled by the server.
    """
    data = []
    with open(server_log_file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _get_random_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
