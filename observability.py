"""Small process-wide backend logger."""
import logging
import os

logging.basicConfig(
    level=os.getenv("SMARTDESK_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def log(name: str) -> logging.Logger:
    """Return a namespaced SmartDesk logger."""
    return logging.getLogger(f"smartdesk.{name}")
