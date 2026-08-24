"""Loopback-only FastAPI surface over :class:`tst.service.TSTService`."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from tst.scope.models import Scope
from tst.service.errors import ServiceError
from tst.service.service import TSTService

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised only without the router extra
    raise RuntimeError("FastAPI is required for `tst serve`; install the router extra") from exc


class ProjectSelectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str = Field(min_length=1, max_length=4_096)


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=100_000)
    budget: int = Field(default=2_000, ge=1, le=1_000_000)


class MemoryCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1, max_length=1_000_000)
    scope: Scope = Scope.PROJECT
    key: str | None = Field(default=None, max_length=512)
    subject: str | None = Field(default=None, max_length=512)
    memory_type: str = Field(default="fact", min_length=1, max_length=128)
    source_text: str | None = Field(default=None, max_length=1_000_000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class MemoryUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1, max_length=1_000_000)
    scope: Scope | None = None
    memory_type: str | None = Field(default=None, max_length=128)
    source_text: str | None = Field(default=None, max_length=1_000_000)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] | None = None


class MemoryMoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_scope: Scope
    source_scope: Scope | None = None


class TreeQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str | int
    depth: int = Field(default=1, ge=0, le=32)
    max_nodes: int = Field(default=40, ge=1, le=10_000)
    token_budget: int = Field(default=2_000, ge=1, le=1_000_000)


def create_app(
    service: TSTService | None = None,
    *,
    project: str | Path | None = None,
    host: str = "127.0.0.1",
) -> FastAPI:
    """Create an app; service construction is deferred until lifespan startup."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("TST's local server only binds loopback addresses")
    owns_service = service is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.service is None:
            app.state.service = TSTService(project)
        try:
            yield
        finally:
            if app.state.owns_service:
                app.state.service.close()
                app.state.service = None

    app = FastAPI(
        title="TST Control Plane",
        description="Local, model-neutral context and memory control plane.",
        version="0.3.0",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.owns_service = owns_service
    app.state.session_token = secrets.token_urlsafe(24)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host_header = request.headers.get("host", "")
        if host_header.startswith("[") and "]" in host_header:
            host_header = host_header[1 : host_header.index("]")]
        else:
            host_header = host_header.split(":", 1)[0]
        host_header = host_header.casefold()
        allowed_hosts = {"127.0.0.1", "localhost", "::1", "testserver"}
        if host_header and host_header not in allowed_hosts:
            return JSONResponse(status_code=403, content={"detail": "loopback access only"})
        origin = request.headers.get("origin")
        if origin:
            origin_host = (urlparse(origin).hostname or "").casefold()
            if origin_host not in allowed_hosts:
                return JSONResponse(status_code=403, content={"detail": "loopback origin only"})
        if os.environ.get("TST_REQUIRE_UI_TOKEN", "").casefold() in {"1", "true", "yes"}:
            if (
                request.url.path.startswith("/api/")
                and request.url.path != "/api/v1/status"
                and request.headers.get("x-tst-session") != app.state.session_token
            ):
                return JSONResponse(status_code=401, content={"detail": "invalid TST session"})
        return await call_next(request)

    def service_for(request: Request) -> TSTService:
        current = request.app.state.service
        if current is None:
            raise HTTPException(status_code=503, detail="TST service is not ready")
        return current

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/status")
    def status(request: Request) -> dict[str, Any]:
        result = call(service_for(request).status)
        result["ui_session_token"] = request.app.state.session_token
        return result

    @app.get("/api/v1/projects")
    def projects(request: Request) -> list[dict[str, str]]:
        return [project.to_dict() for project in service_for(request).projects()]

    @app.post("/api/v1/projects/select")
    def select_project(payload: ProjectSelectRequest, request: Request) -> dict[str, str]:
        selected = call(service_for(request).switch_project, payload.project)
        return selected.to_dict()

    @app.post("/api/v1/context/preview")
    def context_preview(payload: ContextRequest, request: Request) -> dict[str, Any]:
        result = call(service_for(request).context_preview, payload.query, budget=payload.budget)
        return result.model_dump(mode="json")

    @app.post("/api/v1/context/retrieve")
    def context_retrieve(payload: ContextRequest, request: Request) -> dict[str, Any]:
        result = call(service_for(request).retrieve_context, payload.query, budget=payload.budget)
        return result.model_dump(mode="json")

    @app.get("/api/v1/memories")
    def memories(request: Request, scope: Scope | None = None, limit: int = 1_000) -> list[dict[str, Any]]:
        return call(service_for(request).list_memories, scope, limit=limit)

    @app.post("/api/v1/memories")
    def create_memory(payload: MemoryCreateRequest, request: Request) -> dict[str, Any]:
        return call(service_for(request).store_memory, **payload.model_dump())

    @app.get("/api/v1/memories/search")
    def search_memories(request: Request, query: str, scope: Scope | None = None, limit: int = 20):
        result = call(service_for(request).search_memories, query, scope, limit=limit)
        return [item.model_dump(mode="json") for item in result]

    @app.patch("/api/v1/memories/{key:path}")
    def update_memory(key: str, payload: MemoryUpdateRequest, request: Request) -> dict[str, Any]:
        return call(service_for(request).update_memory, key, **payload.model_dump())

    @app.delete("/api/v1/memories/{key:path}")
    def delete_memory(key: str, request: Request, scope: Scope | None = None) -> dict[str, Any]:
        return call(service_for(request).forget_memory, key, scope=scope)

    @app.post("/api/v1/memories/{key:path}/move")
    def move_memory(key: str, payload: MemoryMoveRequest, request: Request) -> dict[str, Any]:
        return call(service_for(request).move_memory, key, payload.target_scope, source_scope=payload.source_scope)

    @app.post("/api/v1/index")
    def index(request: Request) -> dict[str, Any]:
        return call(service_for(request).index_project)

    @app.get("/api/v1/tree/find")
    def tree_find(request: Request, name: str, limit: int = 20):
        return call(service_for(request).find_symbol, name, limit=limit)

    @app.post("/api/v1/tree/query")
    def tree_query(payload: TreeQueryRequest, request: Request):
        return call(
            service_for(request).query_symbol,
            payload.symbol,
            depth=payload.depth,
            max_nodes=payload.max_nodes,
            token_budget=payload.token_budget,
        )

    @app.get("/api/v1/integrations")
    def integrations(request: Request):
        return [item.model_dump(mode="json") for item in service_for(request).integrations()]

    @app.get("/api/v1/events/stream")
    def events_stream(request: Request, follow: bool = False):
        service_instance = service_for(request)

        def body():
            for event in service_instance.events.snapshot():
                yield f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"
            if follow:
                subscriber = service_instance.subscribe_events()
                try:
                    for event in service_instance.events.iter_subscriber(subscriber):
                        yield f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"
                finally:
                    service_instance.events.unsubscribe(subscriber)

        return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    bundled_ui = Path(os.environ["TST_UI_DIST"]) if os.environ.get("TST_UI_DIST") else None
    source_ui = Path(__file__).resolve().parents[2] / "ui" / "dist"
    static_directory = bundled_ui if bundled_ui and bundled_ui.is_dir() else source_ui
    if not static_directory.is_dir():
        static_directory = Path(__file__).with_name("static")
    static_index = static_directory / "index.html"
    if static_index.is_file():
        @app.get("/", include_in_schema=False)
        def index_page():
            return FileResponse(static_index)

        app.mount("/", StaticFiles(directory=static_directory, html=True), name="ui")

    return app


def run(
    *,
    project: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = False,
) -> None:
    """Run uvicorn on loopback; port zero is supported for embedding/tests."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("TST's local server only binds loopback addresses")
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required for `tst serve`; install the router extra") from exc
    application = create_app(project=project, host=host)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(f"http://{host}:{port or 8000}")).start()
    uvicorn.run(application, host=host, port=port or 8000, access_log=False)
