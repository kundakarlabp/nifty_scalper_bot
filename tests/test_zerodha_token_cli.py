from __future__ import annotations

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Add scripts directory to path to import the cli functions
scripts_path = str(Path(__file__).parent.parent / "scripts")
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from zerodha_token_cli import write_env_tokens, get_automated_request_token


def test_write_env_tokens_creates_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    write_env_tokens(env_file, "mock_token_123")
    
    content = env_file.read_text(encoding="utf-8")
    assert "ZERODHA_ACCESS_TOKEN=mock_token_123" in content
    assert "BROKER_ACCESS_TOKEN=mock_token_123" in content
    assert "KITE_ACCESS_TOKEN=mock_token_123" in content


def test_write_env_tokens_updates_existing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SOME_OTHER_VAR=abc\nZERODHA_ACCESS_TOKEN=old_token\nexport BROKER_ACCESS_TOKEN=old_token\n",
        encoding="utf-8"
    )
    
    write_env_tokens(env_file, "new_token_456")
    
    content = env_file.read_text(encoding="utf-8")
    assert "SOME_OTHER_VAR=abc" in content
    assert "ZERODHA_ACCESS_TOKEN=new_token_456" in content
    assert "KITE_ACCESS_TOKEN=new_token_456" in content
    
    # Check if BROKER_ACCESS_TOKEN is updated
    broker_found = False
    for line in content.splitlines():
        if "BROKER_ACCESS_TOKEN=" in line:
            assert "new_token_456" in line
            broker_found = True
    assert broker_found


@patch("requests.Session")
def test_get_automated_request_token_success(mock_session_cls) -> None:
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    
    # 1. Login response
    r1 = MagicMock()
    r1.json.return_value = {"status": "success", "data": {"request_id": "req_123"}}
    
    # 2. 2FA response
    r2 = MagicMock()
    r2.json.return_value = {"status": "success"}
    
    # 3. Redirect response(s)
    r3 = MagicMock()
    r3.status_code = 302
    r3.headers = {"Location": "http://127.0.0.1:8000/?status=success&request_token=my_request_token_abc"}
    
    mock_session.post.side_effect = [r1, r2]
    mock_session.get.return_value = r3
    
    # Use a dummy base32 secret (e.g. "OBQXG43XN5ZGI===")
    token = get_automated_request_token("user", "pass", "OBQXG43XN5ZGI===", "api_key_123")
    assert token == "my_request_token_abc"
