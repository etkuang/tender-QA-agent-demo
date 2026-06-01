# coding: utf-8
# @Author: Wang Qingkang

import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSES = []


def get_free_port() -> int:
    """Get one ephemeral port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def get_distinct_ports(count: int) -> list[int]:
    """Allocate distinct ephemeral ports for multiple services."""
    ports = []
    while len(ports) < count:
        port = get_free_port()
        if port not in ports:
            ports.append(port)
    return ports


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 20.0, interval: float = 0.2) -> bool:
    """Wait until a TCP service starts listening on the given host/port."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(interval)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(interval)
    return False


def build_subprocess_kwargs() -> dict:
    """Build cross-platform process-group options."""
    kwargs = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return kwargs


def stop_all_services(signum, frame):
    """Stop all child processes gracefully, then force-kill if needed."""
    print("\nShutting down services...")
    for proc in PROCESSES:
        try:
            if proc.poll() is not None:
                continue

            if platform.system() == "Windows":
                proc.send_signal(signal.CTRL_C_EVENT)
            else:
                proc.terminate()

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as exc:
            print(f"Failed to stop process {getattr(proc, 'pid', 'unknown')}: {exc}")
    sys.exit(0)


def launch_process(command: list[str], env: dict, kwargs: dict) -> subprocess.Popen:
    process = subprocess.Popen(command, env=env, cwd=PROJECT_ROOT, **kwargs)
    PROCESSES.append(process)
    return process


def main():
    signal.signal(signal.SIGINT, stop_all_services)
    signal.signal(signal.SIGTERM, stop_all_services)

    kwargs = build_subprocess_kwargs()
    agent_port, backend_port, frontend_port = get_distinct_ports(3)

    shared_env = os.environ.copy()
    shared_env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + shared_env.get("PYTHONPATH", "")

    agent_base_url = f"http://127.0.0.1:{agent_port}/v1"
    backend_url = f"http://127.0.0.1:{backend_port}"

    # 1) Start agent layer
    agent_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "agent_layer.interfaces.http.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(agent_port),
    ]
    launch_process(agent_cmd, shared_env.copy(), kwargs)

    if not wait_for_port(agent_port):
        print(f"Agent failed to start on port {agent_port}")
        stop_all_services(None, None)

    # 2) Start backend layer
    backend_env = shared_env.copy()
    backend_env["AGENT_BASE_URL"] = agent_base_url
    backend_env["BACKEND_URL"] = backend_url
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app_backend_layer.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(backend_port),
    ]
    launch_process(backend_cmd, backend_env, kwargs)

    if not wait_for_port(backend_port):
        print(f"Backend failed to start on port {backend_port}")
        stop_all_services(None, None)

    # 3) Start frontend layer
    frontend_env = shared_env.copy()
    frontend_env["AGENT_BASE_URL"] = agent_base_url
    frontend_env["BACKEND_URL"] = backend_url
    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app_frontend_layer/app.py",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(frontend_port),
    ]
    launch_process(frontend_cmd, frontend_env, kwargs)

    if not wait_for_port(frontend_port):
        print(f"Frontend failed to start on port {frontend_port}")
        stop_all_services(None, None)

    print("=" * 72)
    print("Three-layer services started")
    print(f"Agent API:    http://127.0.0.1:{agent_port}")
    print(f"Backend API:  {backend_url}")
    print(f"Frontend UI:  http://127.0.0.1:{frontend_port}")
    print("=" * 72)

    while True:
        for proc in PROCESSES:
            if proc.poll() is not None:
                print(f"Process {proc.pid} exited unexpectedly.")
                stop_all_services(None, None)
        time.sleep(1)


if __name__ == "__main__":
    main()