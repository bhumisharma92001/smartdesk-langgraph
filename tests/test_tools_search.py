import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_exception.exceptions import ToolAuthError  # noqa: E402
from tools.search import web_search  # noqa: E402


def test_web_search_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ToolAuthError):
        web_search.invoke({"query": "langgraph"})


def test_web_search_parses_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    fake_response = Mock()
    fake_response.raise_for_status = Mock()
    fake_response.json.return_value = {
        "results": [{"title": "LangGraph Docs", "url": "https://example.com", "content": "intro"}]
    }
    with patch("tools.search.requests.post", return_value=fake_response):
        result = web_search.invoke({"query": "langgraph", "max_results": 1})

    assert result == [{"title": "LangGraph Docs", "url": "https://example.com", "snippet": "intro"}]