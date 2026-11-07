"""Local content extraction: plain HTTP first, headless Chromium fallback.

Design goals, in order:

1. **No external dependency for the common case.** Most pages are static
   enough that a plain HTTP GET plus HTML-to-text conversion produces the
   same content a paid scraping API would return, at zero cost and with no
   third-party outage or credit exhaustion in the path.
2. **Correct on JavaScript-rendered pages.** When the HTTP fetch yields
   too little text (a shell page), retry through headless Chromium via
   Playwright and take the rendered DOM.
3. **Same safety posture as the paid backends.** Every URL — and every
   post-redirect final URL — passes through the identical
   ``is_safe_url`` / ``check_website_access`` gates the Firecrawl provider
   uses. This provider is not a policy bypass.
4. **Fail per-URL, never per-batch.** A bad URL yields an item with an
   ``error`` field; the rest of the batch still returns.

Config::

    web:
      extract_backend: "local-chrome"

No environment variables. No API key.
"""

from __future__ import annotations

import concurrent.futures as _cf
import logging
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

# A page yielding less text than this after HTTP extraction is treated as a
# JavaScript shell and retried through headless Chromium.
_THIN_CONTENT_CHARS = 600
_HTTP_TIMEOUT = 25
_BROWSER_TIMEOUT_MS = 30_000
_MAX_BYTES = 5_000_000
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_DROP_TAGS = {"script", "style", "noscript", "svg", "canvas", "template", "iframe"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "main", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br", "blockquote", "pre",
}


class _TextExtractor(HTMLParser):
    """Minimal, dependency-free HTML-to-text converter.

    Keeps a heading marker so downstream summarisation retains structure,
    drops script/style/nav noise, and collapses whitespace. Deliberately
    simple: this is a text extractor, not a markdown renderer.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _DROP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self.parts.append("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data.strip()
            return
        if data.strip():
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r" ?\n ?", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_to_text(html: str) -> tuple[str, str]:
    """Return ``(text, title)`` for an HTML document; never raises."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 — malformed HTML is expected in the wild
        logger.debug("HTML parse degraded for a document: %s", exc)
    return parser.text(), parser.title


def _fetch_http(url: str) -> tuple[str, str, str]:
    """Plain HTTP GET. Returns ``(html, final_url, error)``."""
    req = urllib.request.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "html" not in ctype and "text" not in ctype and "json" not in ctype:
                return "", resp.geturl(), f"unsupported content-type: {ctype or 'unknown'}"
            raw = resp.read(_MAX_BYTES)
            charset = resp.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace"), resp.geturl(), ""
    except urllib.error.HTTPError as exc:
        return "", url, f"HTTP {exc.code} {exc.reason}"
    except Exception as exc:  # noqa: BLE001 — network failures are per-URL data
        return "", url, f"fetch failed: {exc}"


def _render_chromium(url: str) -> tuple[str, str, str]:
    """Render *url* in headless Chromium. Returns ``(html, final_url, error)``.

    Uses an ephemeral browser context so no profile, cookie jar, or stored
    credential from the host is ever attached to a fetch.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "", url, "playwright not installed (pip install playwright && playwright install chromium)"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
            try:
                context = browser.new_context(user_agent=_USER_AGENT, java_script_enabled=True)
                page = context.new_page()
                page.goto(url, timeout=_BROWSER_TIMEOUT_MS, wait_until="domcontentloaded")
                try:
                    page.wait_for_load_state("networkidle", timeout=8_000)
                except Exception:  # noqa: BLE001 — networkidle is best-effort
                    pass
                html = page.content()
                final_url = page.url
                context.close()
                return html, final_url, ""
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 — a browser failure is per-URL data
        return "", url, f"headless render failed: {exc}"


def _extract_one(url: str, max_chars: int) -> Dict[str, Any]:
    """Extract a single URL through the policy gates. Never raises."""
    if not is_safe_url(url):
        return {"url": url, "title": "", "content": "", "error": "URL failed safety check"}
    blocked = check_website_access(url)
    if blocked:
        return {
            "url": url, "title": "", "content": "",
            "error": blocked["message"],
            "blocked_by_policy": {
                "host": blocked.get("host"),
                "rule": blocked.get("rule"),
                "source": blocked.get("source"),
            },
        }

    engine = "http"
    html, final_url, error = _fetch_http(url)
    text, title = _html_to_text(html) if html else ("", "")

    if len(text) < _THIN_CONTENT_CHARS:
        # Static fetch produced a shell (or failed) — try the browser.
        r_html, r_final, r_error = _render_chromium(url)
        if r_html:
            r_text, r_title = _html_to_text(r_html)
            if len(r_text) > len(text):
                engine, html, final_url = "chromium", r_html, r_final
                text, title, error = r_text, r_title, ""
        elif not html:
            error = error or r_error

    if not text:
        return {"url": url, "title": title, "content": "",
                "error": error or "no extractable text content"}

    # Re-check the post-redirect destination, matching the paid backends.
    if final_url and final_url != url:
        final_blocked = check_website_access(final_url)
        if final_blocked:
            return {
                "url": url, "title": "", "content": "",
                "error": f"redirect destination blocked: {final_blocked['message']}",
            }

    truncated = False
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    return {
        "url": final_url or url,
        "title": title,
        "content": text,
        # The consuming pipeline (tools/web_tools.py) prefers ``raw_content``
        # when present, so this must be the cleaned text — handing back raw
        # HTML here would push markup into the model's context instead of
        # readable content.
        "raw_content": text,
        "metadata": {
            "engine": engine,
            "requested_url": url,
            "truncated": truncated,
            "chars": len(text),
            "html_bytes": len(html),
        },
    }


class LocalChromeWebProvider(WebSearchProvider):
    """Keyless, cost-free extract provider: HTTP first, headless Chromium fallback."""

    @property
    def name(self) -> str:
        return "local-chrome"

    @property
    def display_name(self) -> str:
        return "Local (HTTP + headless Chromium)"

    def is_available(self) -> bool:
        # The HTTP path uses only the standard library, so this provider is
        # always usable; Playwright only widens what it can render.
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from *urls* locally. Per-URL failures are data, not exceptions."""
        try:
            from tools.interrupt import is_interrupted as _is_interrupted
        except Exception:  # noqa: BLE001 — interrupt support is optional
            def _is_interrupted() -> bool:
                return False

        max_chars = int(kwargs.get("max_chars") or 0)
        results: List[Dict[str, Any]] = []
        for url in urls:
            if _is_interrupted():
                results.append({"url": url, "error": "Interrupted", "title": ""})
                continue
            logger.info("local-chrome extracting: %s", url)
            try:
                # Hard wall-clock bound so one pathological page cannot hang a run.
                with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                    results.append(pool.submit(_extract_one, url, max_chars).result(timeout=90))
            except _cf.TimeoutError:
                results.append({"url": url, "title": "", "content": "",
                                "error": "local extraction exceeded 90s"})
            except Exception as exc:  # noqa: BLE001 — never break the batch
                results.append({"url": url, "title": "", "content": "",
                                "error": f"local extraction failed: {exc}"})
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Local (HTTP + headless Chromium)",
            "badge": "free · no key · extract only",
            "tag": "Extracts locally — plain HTTP, then headless Chromium for JS pages (pair with any search provider)",
            "env_vars": [],
        }
