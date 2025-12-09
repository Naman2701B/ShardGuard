# src/shardguard/registry.py
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REG_MCP_KEY = "mcps"

DEFAULT_HTTP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-06-18",
}

DEFAULT_TOOLS_CONFIG = {
    "allow": ["*"],
    "deny": [],
    "rename": {},
}


def load_registry(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {REG_MCP_KEY: {}}

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if REG_MCP_KEY not in data or not isinstance(data[REG_MCP_KEY], dict):
        data[REG_MCP_KEY] = {}
    return data


def _atomic_write(path: str, data: dict[str, Any]) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + f".tmp.{int(time.time() * 1000)}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    shutil.move(str(tmp), str(p))


def add_mcp(
    registry_path: str,
    *,
    name: str,
    transport: str,
    description: str | None = None,
    http: dict[str, Any] | None = None,
    stdio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reg = load_registry(registry_path)
    mcps = reg[REG_MCP_KEY]

    tnorm = (transport or "").strip().lower()
    if tnorm in {"http", "streaming-http", "stream-http", "streamablehttp"}:
        tnorm = "streamable-http"
    if tnorm not in {"streamable-http", "stdio"}:
        raise ValueError("transport must be one of: streamable-http, stdio")

    entry: dict[str, Any] = {"transport": tnorm}

    if description:
        entry["description"] = description

    if tnorm == "streamable-http":
        if not http or not isinstance(http, dict) or not http.get("url"):
            raise ValueError("http.url is required for streamable-http")

        user_headers = http.get("headers") or {}
        combined_headers = DEFAULT_HTTP_HEADERS.copy()
        combined_headers.update(user_headers)

        entry["http"] = {
            "url": str(http["url"]).rstrip("/"),
            "headers": combined_headers,
        }
    else:
        if not stdio or not isinstance(stdio, dict) or not stdio.get("cmd"):
            raise ValueError("stdio.cmd is required for stdio")
        entry["stdio"] = {
            "cmd": stdio["cmd"],
            **({"args": stdio.get("args")} if stdio.get("args") else {}),
            **({"cwd": stdio.get("cwd")} if stdio.get("cwd") else {}),
            **({"env": stdio.get("env")} if stdio.get("env") else {}),
            "framing": (stdio.get("framing") or "jsonl").lower(),
        }
    entry["tools"] = DEFAULT_TOOLS_CONFIG.copy()
    mcps[name] = entry
    _atomic_write(registry_path, reg)
    return reg


def remove_mcp(registry_path: str, names: Iterable[str]) -> tuple[list[str], list[str]]:
    reg = load_registry(registry_path)
    mcps = reg[REG_MCP_KEY]
    removed, missing = [], []
    for n in names:
        if n in mcps:
            mcps.pop(n, None)
            removed.append(n)
        else:
            missing.append(n)
    _atomic_write(registry_path, reg)
    return removed, missing


def _parse_json_arg(arg: str | None) -> Any | None:
    """Safely parse a JSON string argument, returning None on failure."""
    if not arg:
        return None
    try:
        return json.loads(arg)
    except Exception:
        return None


def parse_transport_config(
    transport: str,
    url: str | None,
    headers: str | None,
    cmd: str | None,
    args: str | None,
    cwd: str | None,
    env: str | None,
    framing: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Helper to parse transport registry
    Raises ValueError if required arguments are missing.
    """
    if transport in {"streamable-http", "http"}:
        if not url:
            raise ValueError("--url required for http")

        http_config = {"url": url}

        parsed_headers = _parse_json_arg(headers)
        if parsed_headers:
            http_config["headers"] = parsed_headers

        return http_config, None

    if not cmd:
        raise ValueError("--cmd required for stdio")

    stdio_config = {"cmd": cmd, "framing": framing}

    parsed_args = _parse_json_arg(args)
    if parsed_args:
        stdio_config["args"] = parsed_args

    if cwd:
        stdio_config["cwd"] = cwd

    parsed_env = _parse_json_arg(env)
    if parsed_env:
        stdio_config["env"] = parsed_env

    return None, stdio_config
