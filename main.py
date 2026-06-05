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

from common.logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSES = []

def get_launcher_logger():
    os.environ.setdefault("LOG_SERVICE_NAME", "launcher")
    return get_logger("launcher")


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


def normalize_proxy_url(proxy_server: str) -> str:
    """Return a proxy URL accepted by httpx."""
    if "://" in proxy_server:
        return proxy_server
    return f"http://{proxy_server}"


def parse_windows_proxy_server(proxy_server: str) -> dict[str, str]:
    """Parse the Windows ProxyServer registry value."""
    proxies = {}
    for item in proxy_server.split(";"):
        if "=" not in item:
            continue
        scheme, address = item.split("=", 1)
        proxies[scheme.lower()] = normalize_proxy_url(address)

    if proxies:
        return proxies

    proxy_url = normalize_proxy_url(proxy_server)
    return {"http": proxy_url, "https": proxy_url}


def read_windows_proxy_server() -> str:
    """Read the current Windows user proxy server."""
    if platform.system() != "Windows":
        return ""

    try:
        import winreg

        registry_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path) as key:
            proxy_enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if not proxy_enable:
                return ""
            proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
            return proxy_server
    except OSError:
        return ""


def env_has_proxy(env: dict) -> bool:
    """Return whether the child process environment already has proxy settings."""
    proxy_names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    return any(env.get(name) for name in proxy_names)


def merge_no_proxy(env: dict, hosts: list[str]) -> None:
    """Merge loopback hosts into NO_PROXY without dropping existing values."""
    current = env.get("NO_PROXY") or env.get("no_proxy") or ""
    values = [value.strip() for value in current.split(",") if value.strip()]

    for host in hosts:
        if host not in values:
            values.append(host)

    merged = ",".join(values)
    env["NO_PROXY"] = merged
    env["no_proxy"] = merged


def configure_proxy_environment(env: dict) -> None:
    """Configure proxy behavior for child services."""
    merge_no_proxy(env, ["127.0.0.1", "localhost", "::1"])

    if env_has_proxy(env):
        return

    proxy_server = read_windows_proxy_server()
    if not proxy_server:
        return

    proxies = parse_windows_proxy_server(proxy_server)
    http_proxy = proxies.get("http")
    https_proxy = proxies.get("https") or http_proxy

    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
    if https_proxy:
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy


def stop_all_services(signum, frame):
    """Stop all child processes gracefully, then force-kill if needed."""
    logger = get_launcher_logger()
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
    os.environ.setdefault("LOG_SERVICE_NAME", "launcher")
    logger = get_launcher_logger()

    signal.signal(signal.SIGINT, stop_all_services)
    signal.signal(signal.SIGTERM, stop_all_services)

    kwargs = build_subprocess_kwargs()
    agent_port, backend_port, frontend_port = get_distinct_ports(3)

    shared_env = os.environ.copy()
    configure_proxy_environment(shared_env)
    shared_env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + shared_env.get("PYTHONPATH", "")

    agent_base_url = f"http://127.0.0.1:{agent_port}/v1"
    backend_url = f"http://127.0.0.1:{backend_port}"

    agent_env = shared_env.copy()
    agent_env["LOG_SERVICE_NAME"] = "agent"
    agent_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "agent_layer.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(agent_port),
    ]
    launch_process(agent_cmd, agent_env, kwargs)

    if not wait_for_port(agent_port):
        logger.error("Agent failed to start | port=%s", agent_port)
        stop_all_services(None, None)

    backend_env = shared_env.copy()
    backend_env["LOG_SERVICE_NAME"] = "backend"
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
        logger.error("Backend failed to start | port=%s", backend_port)
        stop_all_services(None, None)

    frontend_env = shared_env.copy()
    frontend_env["LOG_SERVICE_NAME"] = "frontend"
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
        logger.error("Frontend failed to start | port=%s", frontend_port)
        stop_all_services(None, None)

    logger.info(
        "Three-layer services started | agent_api=http://127.0.0.1:%s | backend_api=%s | frontend_ui=http://127.0.0.1:%s",
        agent_port,
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
