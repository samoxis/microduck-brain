"""JSON-RPC 2.0 prin socket-uri unix, un obiect pe linie (NDJSON).

Formatul vine din ``duck-ipc-proto`` și din ``robotctl``:

* ``robot.move`` este o *notificare* (fără ``id``), continuă, trimisă la 20-50 Hz; robotul
  o lasă să expire (deadman, ~500 ms) dacă nu mai vin comenzi;
* ``robot.stop``, ``robot.health`` sunt *cereri* cu ``id`` și răspuns;
* ``tof.stream`` (pe socket-ul lui ``tofd``) și ``robot.subscribe`` (pe cel al lui
  ``robotd``) transformă conexiunea într-un flux de notificări ``tof.frame`` / ``robot.state``.
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
    """Conexiunea de comenzi către ``robotd``: cereri sincrone + notificări."""

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
            raise OSError("neconectat")
        try:
            self._sock.sendall(_encode(method, params, None))
        except OSError:
            self.close()
            raise

    def call(self, method: str, params: dict | None = None) -> dict:
        if self._sock is None or self._file is None:
            raise OSError("neconectat")
        id_ = self._next_id
        self._next_id += 1
        try:
            self._sock.sendall(_encode(method, params, id_))
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                line = self._file.readline()
                if not line:
                    raise OSError("conexiune închisă de robot")
                msg = json.loads(line)
                if msg.get("id") != id_:
                    continue  # notificări sau răspunsuri vechi
                if "error" in msg and msg["error"]:
                    raise RpcError(f"{method}: {msg['error']}")
                return msg.get("result") or {}
            raise OSError(f"{method}: fără răspuns în {self.timeout}s")
        except (OSError, json.JSONDecodeError):
            self.close()
            raise


class Stream(threading.Thread):
    """Fir care ține o conexiune abonată și apelează ``on_message(method, params)``.

    Se reconectează singur; ``connected`` arată dacă abonamentul e viu.
    """

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
            except (OSError, json.JSONDecodeError) as e:
                log.debug("%s: %s", self.subscribe, e)
            self.connected = False
            self.stop.wait(0.5)

    def _session(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect(self.path)
            s.sendall(_encode(self.subscribe, self.sub_params, 1))
            f = s.makefile("rb")
            while not self.stop.is_set():
                try:
                    line = f.readline()
                except TimeoutError:
                    continue
                if not line:
                    return
                msg = json.loads(line)
                if msg.get("id") == 1:
                    self.answer = msg.get("result") or {}
                    self.connected = True
                    continue
                method = msg.get("method")
                if method and msg.get("params") is not None:
                    self.on_message(method, msg["params"])
        finally:
            s.close()
