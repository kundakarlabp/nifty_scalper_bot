import json
import time
from urllib.error import HTTPError
from urllib.request import urlopen
import pytest

from src.utils import health


def _url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}{path}"


def test_live_and_ready_endpoints() -> None:
    health.stop_health_server()
    health.start_health_server("127.0.0.1", 0)
    assert health._srv is not None  # type: ignore[attr-defined]
    port = health._srv.server_port  # type: ignore[attr-defined]

    with urlopen(_url(port, "/live")) as resp:
        payload = json.loads(resp.read())
    assert payload["status"] == "live"

    with pytest.raises(HTTPError) as exc:
        urlopen(_url(port, "/ready"))
    body = json.loads(exc.value.read())
    assert body["status"] == "not_ready"

    health.STATE.last_tick_ts = time.time()
    health.STATE.broker_connected = True
    with urlopen(_url(port, "/ready")) as resp:
        payload = json.loads(resp.read())
    assert payload["status"] == "ready"

    health.stop_health_server()
