"""Research tools."""
import os
import re
from io import BytesIO
import requests
from langchain_core.tools import ToolException, tool
from observability import log
from tools.schemas import FetchPageInput, WebSearchInput
logger = log("tools.research")


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Tavily when current sourced information is required."""
    logger.info("web_search started | max_results=%d", max_results)
    try:
        key = os.environ["TAVILY_API_KEY"]
        response = requests.post("https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": max_results}, timeout=15)
        response.raise_for_status()
        results = [{"title": item["title"], "url": item["url"],
                    "snippet": item.get("content", "")} for item in response.json()["results"]]
        logger.info("web_search finished | results=%d", len(results)); return results
    except Exception as exc:
        logger.exception("web_search failed")
        raise ToolException(f"Search failed: {exc}") from exc


@tool("fetch_page", args_schema=FetchPageInput)
def fetch_page(url: str) -> str:
    """Fetch and clean HTML or extract text from a PDF research source."""
    logger.info("fetch_page started | url=%s", url)
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        if "pdf" in response.headers.get("content-type", "").lower() or url.lower().endswith(".pdf"):
            from pypdf import PdfReader
            pages = (page.extract_text() or "" for page in PdfReader(BytesIO(response.content)).pages)
            text = "\n".join(pages)[:12000]
            logger.info("fetch_page finished | type=pdf chars=%d", len(text)); return text
        text = re.sub(r"<(script|style).*?</\1>", " ", response.text, flags=re.I | re.S)
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))[:12000]
        logger.info("fetch_page finished | type=html chars=%d", len(text)); return text
    except Exception as exc:
        logger.exception("fetch_page failed | url=%s", url)
        raise ToolException(f"Page fetch failed: {exc}") from exc
