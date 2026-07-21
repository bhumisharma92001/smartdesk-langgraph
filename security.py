"""Process-wide LangGraph deserialization hardening."""
import os
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
