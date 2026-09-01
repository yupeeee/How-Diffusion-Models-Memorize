"""Repository-wide guard against accidental network access in unit tests."""

from __future__ import annotations

import importlib
import socket
import sys
from pathlib import Path
from typing import NoReturn

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_UTILS = (_REPOSITORY_ROOT / "utils").resolve()


def _is_local_utils(module: object) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        Path(str(module_file)).resolve().relative_to(_LOCAL_UTILS)
        return True
    except (OSError, ValueError):
        return False


while str(_REPOSITORY_ROOT) in sys.path:
    sys.path.remove(str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT))
loaded_utils = {
    name: module
    for name, module in tuple(sys.modules.items())
    if name == "utils" or name.startswith("utils.")
}
if any(not _is_local_utils(module) for module in loaded_utils.values()):
    for name in sorted(loaded_utils, reverse=True):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _blocked_network(*arguments: object, **keywords: object) -> NoReturn:
    target = arguments[-1] if arguments else keywords.get("address", "unknown target")
    raise AssertionError(f"Unmocked network access is forbidden in tests: {target}")


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail before an IPv4 or IPv6 socket can contact the network."""

    internet_families = {socket.AF_INET, socket.AF_INET6}
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_send = socket.socket.send
    original_sendall = socket.socket.sendall
    original_sendto = socket.socket.sendto

    def guarded(method: object) -> object:
        def call(
            network_socket: socket.socket, *args: object, **kwargs: object
        ) -> object:
            if network_socket.family in internet_families:
                _blocked_network(*args, **kwargs)
            return method(network_socket, *args, **kwargs)  # type: ignore[operator]

        return call

    monkeypatch.setattr(socket, "create_connection", _blocked_network)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked_network)
    monkeypatch.setattr(socket, "gethostbyname", _blocked_network)
    monkeypatch.setattr(socket, "gethostbyname_ex", _blocked_network)
    monkeypatch.setattr(socket.socket, "connect", guarded(original_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guarded(original_connect_ex))
    monkeypatch.setattr(socket.socket, "send", guarded(original_send))
    monkeypatch.setattr(socket.socket, "sendall", guarded(original_sendall))
    monkeypatch.setattr(socket.socket, "sendto", guarded(original_sendto))
