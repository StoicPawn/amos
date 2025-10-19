"""Very small YAML parser supporting the configuration files used in tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.lower() in {"true", "yes"}:
        return True
    if value.lower() in {"false", "no"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].strip()
        if not items:
            return []
        return [_parse_scalar(part.strip()) for part in items.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result: Dict[str, Any] = {}
        for chunk in inner.split(","):
            key, _, val = chunk.partition(":")
            result[key.strip()] = _parse_scalar(val)
        return result
    return value


def safe_load(stream: str | Path | Any) -> Dict[str, Any]:
    if hasattr(stream, "read"):
        text = stream.read()
    elif isinstance(stream, Path):
        text = stream.read_text(encoding="utf-8")
    else:
        text = str(stream)
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        key, sep, remainder = line.strip().partition(":")
        value = remainder.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not sep or value == "":
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
            continue
        parent[key] = _parse_scalar(value)
    return root


__all__ = ["safe_load"]
