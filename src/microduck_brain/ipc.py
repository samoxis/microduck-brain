"""JSON-RPC 2.0 over unix sockets, one object per line (NDJSON).

The wire format is taken from ``duck-ipc-proto`` and ``robotctl``, not guessed:

* ``robot.move`` is a *notification* (no ``id``), continuous, sent at 20-50 Hz; the robot lets
  it expire (deadman, ~500 ms) when commands stop arriving;
* ``robot.stop`` and ``robot.health`` are *requests* with an ``id`` and a response;
* ``tof.stream`` (on ``tofd``'s socket) and ``robot.subscribe`` (on ``robotd``'s) turn the
  connection into a stream of ``tof.frame`` / ``robot.state`` notifications.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Callable

log = logging.getLogger("microduck_brain.ipc")


class RpcError(Exception):
    pass


def _encode(method: str, params: dict | None, id_: int | None) -> bytes:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if id_ is not None:
        msg["id"] = id_
    if params is not None:
        msg["params"] = params
    return (json.dumps(msg) + "\n").encode()


class Link:
    """The command connection to ``robotd``: synchronous requests + notifications."""

    def __init__(self, path: str, timeout: float = 2.0) -> None:
        self.path = path
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._file = None
        self._next_id = 1

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        self.close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.path)
        except OSError:
            s.close()
            raise
        self._sock = s
        self._file = s.makefile("rb")

    def close(self) -> None:
        for obj in (self._file, self._sock):
            try:
                if obj is not None:
                    obj.close()
            except OSError:
                pass
        self._file = None
        self._sock = None

    def notify(self, method: str, params: dict | None = None) -> None:
        if self._sock is None:
            raise OSError("not connected")
        try:
            self._sock.sendall(_encode(method, params, None))
        except OSError:
            self.close()
            raise

    def call(self, method: str, params: dict | None = None) -> dict:
        if self._sock is None or self._file is None:
            raise OSError("not connected")
        id_ = self._next_id
        self._next_id += 1
        try:
            self._sock.sendall(_encode(method, params, id_))
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                line = self._file.readline()
                if not line:
                    raise OSError("connection closed by the robot")
                msg = json.loads(line)
                if msg.get("id") != id_:
                    continue  # notifications or stale responses
                if "error" in msg and msg["error"]:
                    raise RpcError(f"{method}: {msg['error']}")
                return msg.get("result") or {}
            raise OSError(f"{method}: no response within {self.timeout}s")
        except (OSError, json.JSONDecodeError):
            self.close()
            raise


class Stream(threading.Thread):
    """A thread that keeps a subscribed connection open and calls ``on_message(method, params)``.

    It reconnects on its own; ``connected`` tells whether the subscription is live.
    """

    MAX_LINE_BYTES = 1024 * 1024

    def __init__(
        self,
        path: str,
        subscribe: str,
        sub_params: dict | None,
        on_message: Callable[[str, dict], None],
        stop: threading.Event,
    ) -> None:
        super().__init__(name=f"stream:{subscribe}", daemon=True)
        self.path = path
        self.subscribe = subscribe
        self.sub_params = sub_params
        self.on_message = on_message
        self.stop = stop
        self.connected = False
        self.answer: dict = {}

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self._session()
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                log.debug("%s: %s", self.subscribe, e)
            self.connected = False
            self.stop.wait(0.5)

    def _session(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect(self.path)
            s.sendall(_encode(self.subscribe, self.sub_params, 1))
            # makefile().readline() cannot safely resume after a socket timeout.
            # Keep NDJSON framing here so both idle and partial reads can resume.
            pending = bytearray()
            while not self.stop.is_set():
                try:
                    chunk = s.recv(4096)
                except TimeoutError:
                    continue
                if not chunk:
                    return
                pending.extend(chunk)
                while not self.stop.is_set():
                    end = pending.find(b"\n")
                    if end < 0:
                        if len(pending) > self.MAX_LINE_BYTES:
                            raise OSError("stream message exceeds maximum size")
                        break
                    if end > self.MAX_LINE_BYTES:
                        raise OSError("stream message exceeds maximum size")
                    line = bytes(pending[:end])
                    del pending[:end + 1]
                    msg = json.loads(line)
                    if not isinstance(msg, dict):
                        raise OSError("stream message must be a JSON object")
                    if msg.get("id") == 1:
                        self.answer = msg.get("result") or {}
                        self.connected = True
                        continue
                    method = msg.get("method")
                    if method and msg.get("params") is not None:
                        self.on_message(method, msg["params"])
        finally:
            s.close()
