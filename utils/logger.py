import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured application logger."""
    level = getattr(logging, os.getenv("SMARTDESK_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return logging.getLogger(name)
