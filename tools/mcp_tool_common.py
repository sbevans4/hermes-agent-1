"""Small pure helpers shared by the tools.mcp_tool_* modules: SDK 1.x/2.x field access,
error-text sanitising, numeric/bool coercion, timeouts and jitter. No origin state."""

import logging
import math
import os
import random
import re
from typing import Any, Optional

logger = logging.getLogger("tools.mcp_tool")


class _OriginProxy:
    """Attribute proxy for ``tools.mcp_tool`` resolved at access time. The split modules read
    origin state (``_servers``, ``_lock``, SDK symbols, patchable helpers) through this so
    ``mock.patch("tools.mcp_tool.X")`` and origin-side rebinds stay effective, and so no split
    module needs the origin imported first (the origin imports them while initialising)."""

    __slots__ = ()

    def __getattr__(self, name: str):
        from tools import mcp_tool
        return getattr(mcp_tool, name)


_core = _OriginProxy()
_MISSING = object()


def mcp_field(obj, snake: str, camel: str, default=None):
    """Read an MCP model field across the 1.x -> 2.x rename to snake_case. Pydantic aliases
    don't apply to attribute access, so ``getattr(result, "isError", False)`` silently returns
    the default on 2.x — failed calls read as successful, schemas as empty."""
    value = getattr(obj, snake, _MISSING)
    if value is _MISSING:
        value = getattr(obj, camel, _MISSING)
    return default if value is _MISSING else value


_DEFAULT_TOOL_TIMEOUT = 300      # seconds for tool calls


def _resolve_tool_timeout(config: dict) -> float:
    """Per-server tool-call timeout. Precedence: ``mcp_servers.<name>.timeout`` >
    ``timeouts.mcp.tool_call`` > the 300s default; values are platform-clamped by
    ``resolve_timeout``."""
    per_server = config.get("timeout")
    if per_server is not None:
        return per_server
    try:
        from agent.deadline import resolve_timeout
        resolved = resolve_timeout("mcp.tool_call", default=_DEFAULT_TOOL_TIMEOUT)
        if resolved is not None:
            return resolved
    except Exception:
        logger.debug("mcp.tool_call timeout resolution failed", exc_info=True)
    return _DEFAULT_TOOL_TIMEOUT


# Jitter on reconnect backoff so servers that lost the same backend don't retry in lockstep.
_BACKOFF_JITTER = 0.2            # +/-20%


def _jittered(seconds: float) -> float:
    """``seconds`` with +/-20% uniform jitter, floored at 0."""
    return max(0.0, seconds * random.uniform(1.0 - _BACKOFF_JITTER, 1.0 + _BACKOFF_JITTER))


# Credential patterns to strip from error messages: GitHub PAT, OpenAI-style key, Bearer token,
# and ``token= / key= / API_KEY= / password= / secret=`` assignments.
_CREDENTIAL_PATTERN = re.compile(
    r"(?:ghp_[A-Za-z0-9_]{1,255}|sk-[A-Za-z0-9_-](?:\.?[A-Za-z0-9_-]){0,254}|Bearer\s+\S+"
    r"|(?:token|key|API_KEY|password|secret)=[^\s&,;\"']{1,255})", re.IGNORECASE)


def _env_ref_name(ref: str) -> str:
    """Bare env-var name from a ``${...}`` body; strips a Cursor-style ``env:`` prefix."""
    ref = ref.strip()
    if ref.startswith("env:"):
        ref = ref[len("env:"):].strip()
    return ref


# Scheme runs and authority spans are scanned monotonically, never retried at
# each possible scheme starting letter. No length cap or URL-parser exceptions.
_SCHEME_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+.-")
_ASCII_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def _strip_url_userinfo(text: str) -> str:
    """Strip through the last literal @ in scheme-prefixed authorities.

    Authority ends at whitespace, /, ?, #, <, > or double quote. Apostrophe
    and all RFC 3986 sub-delimiters remain valid userinfo characters. This is
    conservative text sanitization, not URL validation: repeated @, invalid
    percent escapes, Unicode and malformed IPv6 are accepted without parsing.
    Email, protocol-relative URLs, encoded schemes and path/query/fragment @
    are outside this additional rule (legacy credential matching still runs).
    """
    if not isinstance(text, str):
        raise TypeError("error text must be str")
    chunks = []
    cursor = copied = 0
    length = len(text)
    while True:
        marker = text.find("://", cursor)
        if marker < 0:
            break
        # Slash in the previous marker bounds this backward walk. Its span
        # cannot overlap another scheme walk or a consumed authority span.
        start = marker
        while start > cursor and text[start - 1] in _SCHEME_CHARS:
            start -= 1
        while start < marker and text[start] not in _ASCII_LETTERS:
            start += 1
        if start == marker:
            cursor = marker + 3
            continue
        end = marker + 3
        last_at = -1
        while end < length:
            char = text[end]
            if char.isspace() or char in '/?#<>"':
                break
            if char == '@':
                last_at = end
            end += 1
        if last_at >= 0:
            chunks.append(text[copied:marker + 3])
            copied = last_at + 1
        # Visit every marker, including one embedded in this authority.
        # Authority scans stop at the next marker's first slash; backward
        # scheme walks stop at the previous marker. Each span is read O(1)
        # times, without treating RFC userinfo sub-delimiters as boundaries.
        cursor = marker + 3
    if not chunks:
        return text
    chunks.append(text[copied:])
    return "".join(chunks)


def _sanitize_error(text: str) -> str:
    """Remove URL userinfo, then apply the unchanged credential matcher."""
    return _CREDENTIAL_PATTERN.sub("[REDACTED]", _strip_url_userinfo(text))


def _exc_str(exc: BaseException) -> str:
    """Non-empty string for *exc*: some exceptions (``anyio.ClosedResourceError``) carry no
    message, so fall back to ``repr`` to keep diagnostics."""
    text = str(exc).strip()
    return text or repr(exc)


def _prepend_path(env: dict, directory: str) -> dict:
    """Prepend *directory* to env PATH if it is not already present."""
    updated = dict(env or {})
    if directory:
        parts = [part for part in updated.get("PATH", "").split(os.pathsep) if part]
        if directory not in parts:
            parts = [directory, *parts]
        updated["PATH"] = os.pathsep.join(parts) if parts else directory
    return updated


def _safe_numeric(value, default, coerce=int, minimum=1):
    """Coerce a config value (YAML strings included) to a number, clamped to *minimum*;
    *default* on failure or non-finite floats."""
    try:
        result = coerce(value)
        if isinstance(result, float) and not math.isfinite(result):
            return default
        return max(result, minimum)
    except (TypeError, ValueError, OverflowError):
        return default


_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off"})


def _parse_boolish(value: Any, default: bool = True) -> bool:
    """Parse a bool-like config value with safe fallback (YAML ``0``/``1`` are numbers, not words)."""
    if value is None:
        return default
    if isinstance(value, (bool, int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
    logger.warning("MCP config expected a boolean-ish value, got %r; using default=%s", value, default)
    return default


def mcp_server_enabled(cfg: dict) -> bool:
    """Whether ``mcp_servers.<name>`` is on. The ONE reader of the ``enabled`` key: the MCP client,
    the toolset resolver, the profile editor, and every list/status surface call it, so a value
    can never be on for one surface and off for another. Absent, ``null`` or unparseable = on."""
    return _parse_boolish(cfg.get("enabled", True), default=True)


def _get_lifecycle_seconds(config: dict, key: str) -> Optional[float]:
    """Optional positive lifecycle timeout from top-level/nested ``lifecycle`` config (``0``
    disables; negatives and non-numbers are warned about and ignored)."""
    raw = config.get(key)
    if raw is None and isinstance(config.get("lifecycle"), dict):
        raw = config["lifecycle"].get(key)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        logger.warning("MCP config %s must be a number of seconds; ignoring %r", key, raw)
        return None
    if seconds < 0:
        logger.warning("MCP config %s must be positive; ignoring %r", key, raw)
        return None
    return seconds or None
