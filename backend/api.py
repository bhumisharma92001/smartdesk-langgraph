"""FastAPI application exposing the compiled SmartDesk graph."""
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.auth import IdentityStore
from backend.middleware import RequestIDMiddleware
from backend.request_control import RequestController
from graph import build_graph
from memory.checkpointer import build_sqlite_checkpointer, make_thread_config
from memory.context import summarize_history_node
from memory.store import build_persistent_store
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def _maybe_summarize(graph, config: dict, lock) -> None:
    """Compact a thread after its response without racing another turn."""
    with lock:
        try:
            patch = summarize_history_node(graph.get_state(config).values)
            if patch:
                graph.update_state(config, patch)
        except Exception as exc:
            logger.error("Background summarization failed: %s", exc, exc_info=True)


class LoginRequest(BaseModel):
    """Username and an optional returning-browser token."""

    username: str = Field(min_length=1, max_length=50)
    token: str | None = None


class LoginResponse(BaseModel):
    """Durable backend identity returned to the browser."""

    user_id: str
    token: str | None = None


class ChatRequest(BaseModel):
    """One user turn in a new or existing thread."""

    user_id: str
    message: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """Unified SmartDesk response and its durable thread identifier."""

    thread_id: str
    answer: str
    active_agent: str | None


def _public_history(messages: list) -> list[dict[str, str]]:
    """Return one final assistant response for each user turn.

    Specialist handoff messages remain checkpointed for agent context and
    observability, but are implementation detail rather than separate chat
    responses.
    """
    public: list[dict[str, str]] = []
    pending_assistant: AIMessage | None = None
    for message in messages:
        if isinstance(message, HumanMessage):
            if pending_assistant is not None:
                public.append({"role": "assistant", "content": str(pending_assistant.content)})
                pending_assistant = None
            public.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage) and message.content:
            pending_assistant = message
    if pending_assistant is not None:
        public.append({"role": "assistant", "content": str(pending_assistant.content)})
    return public


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create expensive graph resources once and close them cleanly."""
    identities = IdentityStore()
    with build_persistent_store() as store:
        with build_sqlite_checkpointer() as checkpointer:
            app.state.identities = identities
            app.state.requests = RequestController()
            app.state.graph = build_graph(store=store, checkpointer=checkpointer)
            yield
    identities.close()


app = FastAPI(title="SmartDesk API", version="1.0.0", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-User-Token"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled exception request_id=%s: %s", request_id, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers={"X-Request-ID": request_id},
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    """Report API readiness."""
    return {"status": "ok"}


@app.post("/api/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request) -> LoginResponse:
    """Register a username or restore its authenticated browser session."""
    try:
        user_id, token = request.app.state.identities.login(payload.username, payload.token)
        return LoginResponse(user_id=user_id, token=token)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/api/history/{thread_id}")
def history(
    thread_id: str,
    request: Request,
    x_user_token: str = Header(...),
    user_id: str = "",
) -> dict:
    """Return the stored messages for a thread the user owns."""
    identities: IdentityStore = request.app.state.identities
    try:
        identities.authorize(user_id, x_user_token)
        identities.thread(user_id, thread_id)  # verifies ownership
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    config = make_thread_config(thread_id)
    state = request.app.state.graph.get_state(config)
    return {"messages": _public_history(state.values.get("messages") or [])}


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    x_user_token: str = Header(...),
) -> ChatResponse:
    """Run one checkpointed SmartDesk turn for an authenticated user."""
    identities: IdentityStore = request.app.state.identities
    try:
        identities.authorize(payload.user_id, x_user_token)
        thread_id = identities.thread(payload.user_id, payload.thread_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    controls: RequestController = request.app.state.requests
    if not controls.allow(payload.user_id):
        raise HTTPException(status_code=429, detail="Too many requests; please try again shortly")

    lock = controls.thread_lock(thread_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A request for this thread is already running")
    try:
        result = request.app.state.graph.invoke(
            {"messages": [HumanMessage(content=payload.message)], "user_id": payload.user_id},
            config=make_thread_config(thread_id),
        )
        background_tasks.add_task(
            _maybe_summarize,
            request.app.state.graph,
            make_thread_config(thread_id),
            lock,
        )
    finally:
        lock.release()
    return ChatResponse(
        thread_id=thread_id,
        answer=str(result["messages"][-1].content),
        active_agent=result.get("active_agent"),
    )
