"""Small single-origin production proxy for the PixelHolo VM deployment.

The browser receives the Vite build from this process and sends inference
requests to /api.  The proxy streams backend responses instead of buffering
avatar frames or NDJSON events, which keeps the UI's low-latency behavior.
"""

from __future__ import annotations

import os
from pathlib import Path

import aiohttp
from aiohttp import web


STATIC_ROOT = Path(
    os.environ.get("PIXELHOLO_STATIC_ROOT", Path(__file__).resolve().parents[1] / "frontend" / "dist")
).resolve()
BACKEND_ORIGIN = os.environ.get("PIXELHOLO_BACKEND_ORIGIN", "http://127.0.0.1:8000").rstrip("/")


def _forward_headers(headers: aiohttp.typedefs.LooseHeaders) -> dict[str, str]:
    excluded = {"host", "content-length", "connection", "transfer-encoding"}
    return {str(key): str(value) for key, value in headers.items() if str(key).lower() not in excluded}


async def api_proxy(request: web.Request) -> web.StreamResponse:
    if request.method == "OPTIONS":
        return web.Response(status=204)

    backend_path = request.path.removeprefix("/api") or "/"
    url = f"{BACKEND_ORIGIN}{backend_path}"
    if request.query_string:
        url = f"{url}?{request.query_string}"

    body = await request.read()
    session: aiohttp.ClientSession = request.app["http_session"]
    try:
        upstream = await session.request(
            request.method,
            url,
            headers=_forward_headers(request.headers),
            data=body if body else None,
            allow_redirects=False,
        )
    except aiohttp.ClientError as exc:
        raise web.HTTPBadGateway(text=f"Inference backend unavailable: {exc}") from exc

    response = web.StreamResponse(status=upstream.status, headers=_forward_headers(upstream.headers))
    await response.prepare(request)
    try:
        async for chunk in upstream.content.iter_chunked(64 * 1024):
            await response.write(chunk)
    finally:
        upstream.close()
    await response.write_eof()
    return response


async def static_or_spa(request: web.Request) -> web.StreamResponse:
    relative = request.match_info.get("path", "")
    candidate = (STATIC_ROOT / relative).resolve()
    if candidate.is_file() and candidate.is_relative_to(STATIC_ROOT):
        return web.FileResponse(candidate)
    index = STATIC_ROOT / "index.html"
    if not index.is_file():
        raise web.HTTPServiceUnavailable(text=f"Frontend build missing: {index}")
    # The HTML entry point must always be current so a fresh navigation picks
    # up the latest fingerprinted JavaScript bundle after a deployment.
    response = web.FileResponse(index)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


async def on_startup(app: web.Application) -> None:
    app["http_session"] = aiohttp.ClientSession()


async def on_cleanup(app: web.Application) -> None:
    await app["http_session"].close()


app = web.Application(client_max_size=1024**3)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
app.router.add_route("*", "/api/{path:.*}", api_proxy)
app.router.add_route("*", "/{path:.*}", static_or_spa)
