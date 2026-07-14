import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_exception.exceptions import ToolExecutionError 
from tools.fetch_page import fetch_page  


def test_fetch_page_strips_html_to_text():
    fake_response = Mock()
    fake_response.raise_for_status = Mock()
    fake_response.text = "<html><body><script>bad()</script><p>Hello world</p></body></html>"
    with patch("tools.fetch_page.requests.get", return_value=fake_response):
        result = fetch_page.invoke({"url": "https://example.com"})

    assert result == "Hello world"


def test_fetch_page_raises_on_request_failure():
    import requests

    with patch("tools.fetch_page.requests.get", side_effect=requests.RequestException("timeout")):
        with pytest.raises(ToolExecutionError):
            fetch_page.invoke({"url": "https://example.com"})