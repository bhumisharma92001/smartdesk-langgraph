import os

import requests
from langchain_core.tools import tool

from custom_exception.exceptions import ToolAuthError, ToolExecutionError
from tools.schemas import WebSearchInput

_TAVILY_URL = "https://api.tavily.com/search"


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily and return the top matching results.

    Use this when a task requires current or factual information that
    is not already present in the conversation, e.g. news, documentation,
    or general research queries. Follow up with fetch_page on a promising
    URL if you need the full page content rather than just a snippet.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (1-20).

    Returns:
        A list of dicts, each with 'title', 'url', and 'snippet' keys.

    Raises:
        ToolAuthError: If the TAVILY_API_KEY environment variable is not set.
        ToolExecutionError: If the request to Tavily fails.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ToolAuthError("TAVILY_API_KEY environment variable is not set.")

    try:
        resp = requests.post(
            _TAVILY_URL,
            json={"query": query, "max_results": max_results},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ToolExecutionError(f"web_search failed: {exc}") from exc

    return [
        {"title": r["title"], "url": r["url"], "snippet": r.get("content", "")}
        for r in resp.json().get("results", [])
    ]