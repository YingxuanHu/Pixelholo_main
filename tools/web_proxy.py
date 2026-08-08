"""Small single-origin production proxy for the PixelHolo VM deployment.

The browser receives the Vite build from this process and sends inference
requests to /api.  The proxy streams backend responses instead of buffering
avatar frames or NDJSON events, which keeps the UI's low-latency behavior.
"""

from __future__ import annotations

import asyncio
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

    response_headers = _forward_headers(upstream.headers)
    # Cloudflare should forward this response as a live media stream.  These
    # headers also prevent intermediary transformations from accumulating
    # binary PHS1 packets before the browser receives them.
    response_headers.update(
        {
            "Cache-Control": "no-store, no-transform",
            "X-Accel-Buffering": "no",
        }
    )
    response = web.StreamResponse(status=upstream.status, headers=response_headers)
    await response.prepare(request)
    client_closed = False
    try:
        # 64 KiB made the public path wait for several media frames before
        # forwarding.  Smaller writes preserve the backend's packet cadence
        # without changing the binary protocol.
        async for chunk in upstream.content.iter_chunked(16 * 1024):
            if request.transport is None or request.transport.is_closing():
                client_closed = True
                break
            await response.write(chunk)
    except (ConnectionResetError, aiohttp.ClientConnectionError):
        # A browser AbortController or page navigation is normal for an
        # interruptible avatar stream.  Closing the upstream response here
        # propagates the cancellation to FastAPI instead of leaving a stale
        # renderer holding the shared profile lock until a socket timeout.
        client_closed = True
    except asyncio.CancelledError:
        client_closed = True
        raise
    finally:
        upstream.close()
    if not client_closed:
        try:
            await response.write_eof()
        except (ConnectionResetError, aiohttp.ClientConnectionError):
            pass
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
    app["http_session"] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
    )


async def on_cleanup(app: web.Application) -> None:
    await app["http_session"].close()


app = web.Application(client_max_size=1024**3)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)
app.router.add_route("*", "/api/{path:.*}", api_proxy)
app.router.add_route("*", "/{path:.*}", static_or_spa)
