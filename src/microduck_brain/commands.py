"""Commands from outside: a tiny JSON-RPC server on a local unix socket, and a client.

    {"jsonrpc": "2.0", "id": 1, "method": "brain.command", "params": {"name": "stop"}}
    {"jsonrpc": "2.0", "id": 2, "method": "brain.status"}

Names: ``stop`` (stand still until ``go``), ``go``, ``home`` (go back to the start, with [home] on),
``toggle``. The socket is created readable and writable by this user only. A command is queued and
handed to the brain on its next tick (``Inputs.command``), which records it, so a run with commands
replays exactly. Anything that can open the socket can drive the robot: that is the point (a phone
app, a voice assistant, a script) and why it is off by default.

    microduck-brain-cmd stop          # the client, from a shell on the robot
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import threading
from collections import deque
from pathlib import Path

log = logging.getLogger("microduck_brain.commands")

NAMES = ("stop", "go", "home", "toggle")


def default_socket(path: str = "") -> str:
    if path:
        return path
    run = os.environ.get("XDG_RUNTIME_DIR")
    if run and os.path.isdir(run):
        return str(Path(run) / "microduck-brain.sock")
    return str(Path(tempfile.gettempdir()) / f"microduck-brain-{os.getuid()}.sock")


class CommandServer(threading.Thread):
    MAX_LINE = 4096

    def __init__(self, path: str, stop: threading.Event, status=lambda: {}) -> None:
        super().__init__(name="commands", daemon=True)
        self.path = path
        self.stop = stop
        self.status = status                # () -> dict, for brain.status
        self.queue: deque[str] = deque(maxlen=8)
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self.ready = threading.Event()

    def pop(self) -> str | None:
        with self._lock:
            return self.queue.popleft() if self.queue else None

    def _bind(self) -> socket.socket:
        p = Path(self.path)
        if p.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(self.path)
                probe.close()
                raise OSError(f"{self.path}: another microduck-brain is already listening")
            except (ConnectionRefusedError, FileNotFoundError):
                p.unlink()                  # a stale socket from a run that was killed
            finally:
                probe.close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)               # created 0600: this user only
        try:
            s.bind(self.path)
        finally:
            os.umask(old)
        s.listen(4)
        s.settimeout(0.5)
        return s

    def run(self) -> None:
        try:
            self._sock = self._bind()
        except OSError as e:
            log.warning("commands: not listening (%s)", e)
            self.ready.set()
            return
        log.info("commands: listening on %s", self.path)
        self.ready.set()
        try:
            while not self.stop.is_set():
                try:
                    conn, _ = self._sock.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                with conn:
                    conn.settimeout(1.0)
                    try:
                        self._serve(conn)
                    except (OSError, ValueError) as e:
                        log.debug("commands: %s", e)
        finally:
            self._sock.close()
            try:
                os.unlink(self.path)
            except OSError:
                pass

    def _serve(self, conn: socket.socket) -> None:
        buf = b""
        while not self.stop.is_set():
            chunk = conn.recv(1024)
            if not chunk:
                return
            buf += chunk
            if len(buf) > self.MAX_LINE:
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    conn.sendall((json.dumps(self._answer(line)) + "\n").encode())

    def _answer(self, line: bytes) -> dict:
        try:
            msg = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if not isinstance(msg, dict):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "not an object"}}
        id_, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if method == "brain.status":
            return {"jsonrpc": "2.0", "id": id_, "result": self.status()}
        if method == "brain.command":
            name = params.get("name") if isinstance(params, dict) else None
            if name not in NAMES:
                return {"jsonrpc": "2.0", "id": id_,
                        "error": {"code": -32602, "message": f"name must be one of {', '.join(NAMES)}"}}
            with self._lock:
                self.queue.append(name)
            log.info("commands: %s", name)
            return {"jsonrpc": "2.0", "id": id_, "result": {"queued": name}}
        return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32601, "message": f"no method {method!r}"}}


def send(path: str, method: str, params: dict | None = None, timeout: float = 2.0) -> dict:
    """One request to a running brain; its result, or raises OSError / RuntimeError."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        msg = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            msg["params"] = params
        s.sendall((json.dumps(msg) + "\n").encode())
        f = s.makefile("rb")
        reply = json.loads(f.readline() or b"{}")
    finally:
        s.close()
    if reply.get("error"):
        raise RuntimeError(reply["error"].get("message", "error"))
    return reply.get("result") or {}
