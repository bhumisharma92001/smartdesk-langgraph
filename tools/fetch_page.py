import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from typing import Callable, Optional
import trafilatura

from custom_exception.exceptions import ToolExecutionError, ToolInputError
from tools.schemas import FetchPageInput

_HEADERS = {"User-Agent": "SmartDesk/1.0"}


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolInputError(f"Only http/https URLs are allowed, got: {url!r}")


def _fetch_html(url: str) -> str:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ToolExecutionError(f"fetch_page failed for {url!r}: {exc}") from exc
    return resp.text


def _extract_text(html: str, max_chars: int = 3000) -> str:
    """Extract main article content, stripping nav/ads/boilerplate."""
    extracted = trafilatura.extract(html)
    if not extracted:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        extracted = " ".join(soup.get_text(separator=" ").split())

    if len(extracted) > max_chars:
        extracted = extracted[:max_chars] + "... [truncated]"
    return extracted


@tool("fetch_page", args_schema=FetchPageInput)
def fetch_page(url: str, max_chars: int = 3000) -> str:
    """Fetch a URL and return its cleaned, truncated text content.

    Use this after web_search to read the full content of a promising
    result, since search only returns short snippets.

    Args:
        url: The http/https URL to fetch.
        max_chars: Maximum characters of extracted text to return (default 3000).

    Returns:
        Plain text extracted from the page's main content, truncated to max_chars.

    Raises:
        ToolInputError: If the URL scheme is not http/https.
        ToolExecutionError: If the request fails.
    """
    _validate_url(url)
    html = _fetch_html(url)
    return _extract_text(html, max_chars=max_chars)