import os
from typing import TypedDict

import requests
from langchain_core.tools import tool

from custom_exception.exceptions import ToolAuthError, ToolExecutionError
from tools.schemas import WebSearchInput

_TAVILY_URL = "https://api.tavily.com/search"
_MAX_SNIPPET_CHARS = 500


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web via Tavily and return the top matching results.

    Use this when a task requires current or factual information that
    is not already present in the conversation. The returned snippets
    and URLs are the complete research context for the agent.

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
        data = resp.json()
        return [
            {
                "title": r["title"],
                "url": r["url"],
                "snippet": r.get("content", "")[:_MAX_SNIPPET_CHARS],
            }
            for r in data.get("results", [])
        ]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise ToolExecutionError(f"web_search failed: {exc}") from exc
