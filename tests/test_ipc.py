"""Stream regression tests: disposable socket pairs only, never a robot socket."""

import json
import socket
import threading

import pytest

from microduck_brain import ipc


ACK = b'{"id":1,"result":{"accepted":true}}\n'


def frame(seq):
    return json.dumps({"method": "tof.frame", "params": {"seq": seq}}).encode() + b"\n"


@pytest.fixture
def live_session(monkeypatch):
    reader, writer = socket.socketpair()
    timed_out = threading.Event()
    received = threading.Event()
    messages, errors = [], []

    class LocalSocket:
        closed = False
        legacy_file = None

        def settimeout(self, timeout):
            reader.settimeout(0.05)  # exercise the actual timeout without a one-second sleep

        def connect(self, path):
            assert path == "test-only.sock"  # the already-connected pair is the entire transport

        def sendall(self, data):
            reader.sendall(data)

        def makefile(self, mode):
            self.legacy_file = reader.makefile(mode)
            return self

        def readline(self):
            try:
                return self.legacy_file.readline()
            except TimeoutError:
                timed_out.set()
                raise

        def recv(self, size):
            try:
                return reader.recv(size)
            except TimeoutError:
                timed_out.set()
                raise

        def close(self):
            self.closed = True
            if self.legacy_file is not None:
                self.legacy_file.close()
            reader.close()

    transport = LocalSocket()
    monkeypatch.setattr(ipc.socket, "socket", lambda *args: transport)
    stop = threading.Event()

    def on_message(method, params):
        messages.append((method, params))
        received.set()

    stream = ipc.Stream("test-only.sock", "tof.stream", None, on_message, stop)

    def run():
        try:
            stream._session()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    writer.settimeout(2)
    subscription = b""
    while not subscription.endswith(b"\n"):
        subscription += writer.recv(4096)
    assert json.loads(subscription)["method"] == "tof.stream"
    try:
        yield stream, writer, timed_out, received, messages, thread, errors, transport
    finally:
        stop.set()
        writer.close()
        thread.join(2)
        assert not thread.is_alive()
        assert not errors
        assert transport.closed


def test_idle_timeout_keeps_subscription_usable(live_session):
    stream, writer, timed_out, received, messages, *_ = live_session
    writer.sendall(ACK)
    assert timed_out.wait(2)
    writer.sendall(frame(1))
    assert received.wait(2)
    assert stream.connected
    assert stream.answer == {"accepted": True}
    assert messages == [("tof.frame", {"seq": 1})]


def test_fragmented_utf8_message_survives_timeout(live_session):
    _, writer, timed_out, received, messages, *_ = live_session
    message = json.dumps({"method": "tof.frame", "params": {"label": "rață"}}, ensure_ascii=False).encode() + b"\n"
    # Split inside a multi-byte UTF-8 character, before any complete JSON line.
    split = message.index("ț".encode()) + 1
    writer.sendall(message[:split])
    assert timed_out.wait(2)
    writer.sendall(message[split:])
    assert received.wait(2)
    assert messages == [("tof.frame", {"label": "rață"})]


def test_stop_during_idle_exits_and_closes_socket(live_session):
    stream, _, timed_out, _, _, thread, errors, transport = live_session
    assert timed_out.wait(2)
    stream.stop.set()
    thread.join(2)
    assert not thread.is_alive()
    assert transport.closed
    assert not errors


class ScriptedSocket:
    """Deterministic wire chunks for boundaries and reconnection, no network."""

    def __init__(self, *chunks):
        self.chunks = iter(chunks)
        self.closed = False
        self.sent = []

    def settimeout(self, timeout):
        pass

    def connect(self, path):
        assert path == "test-only.sock"

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, size):
        chunk = next(self.chunks, b"")
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def close(self):
        self.closed = True


def make_stream(monkeypatch, sockets, callback):
    factory = iter(sockets)
    monkeypatch.setattr(ipc.socket, "socket", lambda *args: next(factory))
    return ipc.Stream("test-only.sock", "tof.stream", None, callback, threading.Event())


def test_coalesced_lines_and_partial_tail_keep_order(monkeypatch):
    wire = ScriptedSocket(ACK + frame(1) + frame(2)[:9], TimeoutError(), frame(2)[9:])
    messages = []
    stream = make_stream(monkeypatch, [wire], lambda method, params: messages.append(params))
    stream._session()
    assert messages == [{"seq": 1}, {"seq": 2}]
    assert stream.answer == {"accepted": True}
    assert len(wire.sent) == 1
    assert wire.closed


@pytest.mark.parametrize("bad", [b"not-json\n", b"\xff\n", b"[]\n", b"x" * 33, b"x" * 33 + b"\n", OSError("disconnected")])
def test_bad_session_reconnects_and_resubscribes(monkeypatch, bad):
    first = ScriptedSocket(bad)
    second = ScriptedSocket(b'{"method":"ok","params":{}}\n')
    messages = []

    def on_message(method, params):
        messages.append(params)
        stream.stop.set()

    stream = make_stream(monkeypatch, [first, second], on_message)
    stream.MAX_LINE_BYTES = 32
    stream.run()
    assert messages == [{}]
    assert first.closed and second.closed
    assert len(first.sent) == len(second.sent) == 1
    assert json.loads(second.sent[0])["method"] == "tof.stream"
    assert not stream.connected


def test_eof_discards_partial_message_before_reconnect(monkeypatch):
    first = ScriptedSocket(b'{"method":"stale', b"")
    second = ScriptedSocket(ACK + frame(3))
    messages = []

    def on_message(method, params):
        messages.append(params)
        stream.stop.set()

    stream = make_stream(monkeypatch, [first, second], on_message)
    stream.run()
    assert messages == [{"seq": 3}]
    assert first.closed and second.closed


def test_line_at_size_limit_is_accepted(monkeypatch):
    line = frame(4)
    wire = ScriptedSocket(line[:-1], line[-1:])
    messages = []
    stream = make_stream(monkeypatch, [wire], lambda method, params: messages.append(params))
    stream.MAX_LINE_BYTES = len(line) - 1
    stream._session()
    assert messages == [{"seq": 4}]
