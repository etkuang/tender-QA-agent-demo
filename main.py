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

from common.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSES = []

os.environ.setdefault("LOG_SERVICE_NAME", "launcher")
if os.environ.get("DEBUG_MODE", "false").lower() == "true":
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
logger = get_logger("launcher")


def get_free_port() -> int:
    """Get one ephemeral port from the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


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
    logger.info("Shutting down services")

    running_processes = [proc for proc in PROCESSES if proc.poll() is None]

    for proc in running_processes:
        try:
            if platform.system() == "Windows":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except Exception:
            logger.exception("Failed to request process stop | pid=%s", getattr(proc, "pid", "unknown"))

    deadline = time.time() + 15.0
    for proc in running_processes:
        if proc.poll() is not None:
            continue

        remaining = max(0.0, deadline - time.time())
        try:
            # Wait until this subprocess exits, but no longer than remaining seconds
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not exit gracefully; killing | pid=%s", proc.pid)
            proc.kill()
        except Exception:
            logger.exception("Failed to wait for process | pid=%s", getattr(proc, "pid", "unknown"))

    sys.exit(0)


def launch_process(command: list[str], env: dict, kwargs: dict) -> subprocess.Popen:
    process = subprocess.Popen(command, env=env, cwd=PROJECT_ROOT, **kwargs)
    PROCESSES.append(process)
    return process


def main():
    # register stop_all_services as the shutdown handler for the main launcher process
    signal.signal(signal.SIGINT, stop_all_services)
    signal.signal(signal.SIGTERM, stop_all_services)

    kwargs = build_subprocess_kwargs()

    shared_env = os.environ.copy()
    # add the project root to PYTHONPATH
    shared_env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + shared_env.get("PYTHONPATH", "")

    knowledge_base_port = get_free_port()
    knowledge_base_url = f"http://127.0.0.1:{knowledge_base_port}"

    knowledge_base_env = shared_env.copy()
    knowledge_base_env["LOG_SERVICE_NAME"] = "knowledge-base"
    knowledge_base_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "knowledge_base_layer.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(knowledge_base_port),
    ]
    launch_process(knowledge_base_cmd, knowledge_base_env, kwargs)

    if not wait_for_port(knowledge_base_port):
        logger.error("Knowledge-base service failed to start | port=%s", knowledge_base_port)
        stop_all_services(None, None)

    agent_port = get_free_port()
    agent_base_url = f"http://127.0.0.1:{agent_port}"

    agent_env = shared_env.copy()
    agent_env["LOG_SERVICE_NAME"] = "agent"
    agent_env["KNOWLEDGE_BASE_URL"] = knowledge_base_url
    agent_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "agent_layer_old.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(agent_port),
    ]
    launch_process(agent_cmd, agent_env, kwargs)

    if not wait_for_port(agent_port):
        logger.error("Agent failed to start | port=%s", agent_port)
        stop_all_services(None, None)

    backend_port = get_free_port()
    backend_url = f"http://127.0.0.1:{backend_port}"

    backend_env = shared_env.copy()
    backend_env["LOG_SERVICE_NAME"] = "backend"
    backend_env["AGENT_BASE_URL"] = agent_base_url
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
        logger.error("Backend failed to start | port=%s", backend_port)
        stop_all_services(None, None)

    frontend_port = get_free_port()

    frontend_env = shared_env.copy()
    frontend_env["LOG_SERVICE_NAME"] = "frontend"
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
        logger.error("Frontend failed to start | port=%s", frontend_port)
        stop_all_services(None, None)

    logger.info(
        "Services started | knowledge_base_api=%s | agent_api=%s | backend_api=%s | frontend_ui=http://127.0.0.1:%s",
        knowledge_base_url,
        agent_base_url,
        backend_url,
        frontend_port,
    )

    while True:
        for proc in PROCESSES:
            if proc.poll() is not None:
                logger.error("Process exited unexpectedly | pid=%s | returncode=%s", proc.pid, proc.returncode)
                stop_all_services(None, None)
        time.sleep(1)


if __name__ == "__main__":
    main()
