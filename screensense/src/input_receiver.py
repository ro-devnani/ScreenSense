"""
Background socket listener for the ESP32-C3 button receiver.

The ESP pushes JSON lines of the form `{"erase": true, "write": false}\n` over
TCP. A daemon thread keeps the latest values in a lock-guarded dict so the
main tracker loop can poll without blocking.

Public API:
    start_background_listener()
    is_erase_pressed()  -> bool
    is_write_pressed()  -> bool
"""

import socket
import json
import threading
import time


HOST = '0.0.0.0'   # Listen on all network interfaces
PORT = 65432       # Must match the port configured on the ESP32-C3

_button_states = {
    "erase": False,
    "write": False,
}
_state_lock = threading.Lock()


def _internal_socket_worker():
    """Background worker — accepts connections from the ESP and parses JSON
    lines into the shared button-state dict."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[InputReceiver] Listening on port {PORT}")

        while True:
            try:
                conn, addr = server_socket.accept()
                buffer = ""
                with conn:
                    while True:
                        data = conn.recv(1024).decode('utf-8')
                        if not data:
                            break  # Client disconnected; wait for reconnect.

                        buffer += data
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if not line.strip():
                                continue
                            try:
                                incoming = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            with _state_lock:
                                _button_states["erase"] = bool(incoming.get("erase", False))
                                _button_states["write"] = bool(incoming.get("write", False))
            except Exception:
                # Brief pause on connection errors before retrying so a hot
                # loop can't peg the CPU when the network is misbehaving.
                time.sleep(1)


def start_background_listener():
    """Call once at startup — spawns the daemon thread that maintains the
    button states. Safe to call multiple times (subsequent calls no-op)."""
    if getattr(start_background_listener, "_started", False):
        return
    threading.Thread(target=_internal_socket_worker, daemon=True).start()
    start_background_listener._started = True
    print("[InputReceiver] Background thread started.")


def is_erase_pressed() -> bool:
    with _state_lock:
        return _button_states["erase"]


def is_write_pressed() -> bool:
    with _state_lock:
        return _button_states["write"]
