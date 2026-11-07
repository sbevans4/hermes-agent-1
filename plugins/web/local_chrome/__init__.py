"""Local web extraction plugin — bundled, auto-loaded.

Extracts page content with no external API and no per-call cost: a plain
HTTP fetch first, falling back to headless Chromium (Playwright) when the
page needs JavaScript to render. Pairs with any search-only provider
(e.g. ddgs) to give a fully keyless web stack.
"""

from __future__ import annotations

from plugins.web.local_chrome.provider import LocalChromeWebProvider


def register(ctx) -> None:
    """Register the local-chrome provider with the plugin context."""
    ctx.register_web_search_provider(LocalChromeWebProvider())
