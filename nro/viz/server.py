"""Serve the browser viewer and its data API over HTTP.

The server uses only the standard library. Every request must carry the
user's access token: as a ``token`` query parameter, which sets a cookie; as
that cookie; or, for command-line clients, in an ``Authorization: Bearer``
header. Requests select a branch with a ``branch`` query parameter. Files are
addressed by catalog identifiers, so the API cannot read paths outside the
indexed derivatives.
"""

from __future__ import annotations

import gzip
import hmac
import json
import secrets
import sys
import traceback
from collections.abc import Callable
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlencode, urlsplit

from nro.viz import data
from nro.viz.catalog import HEMISPHERES, Catalogs

COOKIE = "nro_viz_token"
STATIC_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "viz.css": "text/css; charset=utf-8",
    "viz.js": "text/javascript; charset=utf-8",
}
COMPRESS_MINIMUM_BYTES = 64 * 1024


class RequestError(Exception):
    """A client error reported to the browser as JSON."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        """Record the HTTP status sent with the message."""
        super().__init__(message)
        self.status = status


class VizServer(ThreadingHTTPServer):
    """Hold the catalogs, token, and seed-correlation state shared by handlers."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        catalogs: Catalogs,
        token: str,
        *,
        restart: Callable[[], None] | None = None,
        cache_bytes: int = 8 * 1024**3,
    ) -> None:
        """Bind the address; port 0 selects a free port.

        ``restart`` is called after the restart response is sent; it must not
        block, because it runs on a request thread. ``cache_bytes`` bounds the
        memory used by normalized time series kept for seed correlation.
        """
        super().__init__(address, VizHandler)
        self.catalogs = catalogs
        self.token = token
        self.correlation = data.SeedCorrelation(cache_bytes)
        self.restart = restart


class VizHandler(BaseHTTPRequestHandler):
    """Authorize each GET request and dispatch it to a static file or API route."""

    server: VizServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Suppress per-request access logs; failures are reported separately."""
        return

    def do_GET(self) -> None:  # noqa: N802
        """Answer one read request."""
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        """Answer one request that changes server state."""
        self._handle("POST")

    def _handle(self, method: str) -> None:
        """Dispatch a request, converting failures into JSON error responses."""
        url = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        try:
            if method == "POST":
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if method == "GET" and url.path == "/" and "token" in query:
                self._login(query)
                return
            self._authorize()
            if method == "POST":
                self._action(url.path, query)
            else:
                self._route(url.path, query)
        except RequestError as error:
            self._json({"error": str(error)}, status=error.status)
        except (KeyError, IndexError, ValueError) as error:
            self._json({"error": str(error).strip("'\"")}, status=HTTPStatus.BAD_REQUEST)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:  # noqa: BLE001 - report unexpected failures to the client
            traceback.print_exc(file=sys.stderr)
            self._json(
                {"error": f"{type(error).__name__}: {error}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _login(self, query: dict[str, str]) -> None:
        if not hmac.compare_digest(query["token"], self.server.token):
            raise RequestError(HTTPStatus.FORBIDDEN, "Invalid access token")
        remaining = {key: value for key, value in query.items() if key != "token"}
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/" + (f"?{urlencode(remaining)}" if remaining else ""))
        self.send_header(
            "Set-Cookie", f"{COOKIE}={self.server.token}; Path=/; HttpOnly; SameSite=Strict"
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorize(self) -> None:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        supplied = cookie[COOKIE].value if COOKIE in cookie else ""
        authorization = self.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            supplied = authorization.removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, self.server.token):
            raise RequestError(
                HTTPStatus.FORBIDDEN,
                "Open the URL printed by `nro viz`, including its token parameter",
            )

    def _route(self, path: str, query: dict[str, str]) -> None:
        catalogs = self.server.catalogs
        branch = query.get("branch", "")
        if path == "/":
            self._static("index.html")
        elif path.startswith("/static/"):
            self._static(path.removeprefix("/static/"))
        elif path == "/api/ping":
            self._json({"ok": True})
        elif path == "/api/branches":
            from nro.viz.catalog import branch_names

            self._json(branch_names())
        elif path == "/api/participants":
            self._json(catalogs.get(branch).participants())
        elif path == "/api/file":
            try:
                layer = catalogs.open_file(query["project"], query["path"], branch=branch)
            except PermissionError as error:
                raise RequestError(HTTPStatus.FORBIDDEN, str(error)) from None
            except FileNotFoundError as error:
                raise RequestError(HTTPStatus.NOT_FOUND, str(error)) from None
            self._json(layer.public())
        elif path == "/api/view":
            view = catalogs.get(branch).view(query["project"], query["participant"])
            self._json(view.public())
        elif path == "/api/mesh":
            surface = catalogs.surface(query["surface"])
            hemisphere, kind = query["hemi"], query["kind"]
            coordinates, triangles = data.mesh(surface.paths[(hemisphere, kind)])
            self._binary(
                coordinates.tobytes() + triangles.tobytes(),
                {"X-Vertices": len(coordinates), "X-Triangles": len(triangles)},
            )
        elif path == "/api/layer":
            layer = catalogs.layer(query["layer"])
            values = data.layer_map(layer, int(query.get("map", 0)))
            self._values(values, layer.vertex_counts)
        elif path == "/api/labels":
            layer = catalogs.layer(query["layer"])
            self._json(data.label_table(layer, int(query.get("map", 0))))
        elif path == "/api/prepare":
            layer = self._connectivity(query["layer"])
            self.server.correlation.prepare(layer.paths["LR"])
            self._json({"ready": True})
        elif path == "/api/seed":
            layer = self._connectivity(query["layer"])
            hemisphere = query["hemi"]
            if hemisphere not in HEMISPHERES:
                raise RequestError(HTTPStatus.BAD_REQUEST, f"Unknown hemisphere {hemisphere!r}")
            values = self.server.correlation.correlate(layer, hemisphere, int(query["vertex"]))
            self._values(values, layer.vertex_counts)
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, f"Unknown path {path}")

    def _action(self, path: str, query: dict[str, str]) -> None:
        if path == "/api/refresh":
            catalog = self.server.catalogs.refresh(query.get("branch", ""))
            self._json({"participants": len(catalog.participants())})
        elif path == "/api/restart":
            if self.server.restart is None:
                raise RequestError(HTTPStatus.NOT_IMPLEMENTED, "This server cannot restart itself")
            self._json({"restarting": True})
            self.wfile.flush()
            self.server.restart()
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, f"Unknown action {path}")

    def _connectivity(self, identifier: str):
        layer = self.server.catalogs.layer(identifier)
        if layer.kind != "connectivity":
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{layer.label} is not a dense time series")
        return layer

    def _static(self, name: str) -> None:
        if name not in STATIC_TYPES:
            raise RequestError(HTTPStatus.NOT_FOUND, f"Unknown static file {name}")
        body = resources.files("nro.viz").joinpath("static", name).read_bytes()
        self._send(body, STATIC_TYPES[name], cache=False)

    def _values(self, values, counts) -> None:
        body, sizes = data.concatenated(values, counts)
        self._binary(body, {"X-Counts": json.dumps(sizes, separators=(",", ":"))})

    def _binary(self, body: bytes, headers: dict[str, object]) -> None:
        self._send(body, "application/octet-stream", headers=headers)

    def _json(self, value: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(value).encode(), "application/json", status=status)

    def _send(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, object] | None = None,
        cache: bool = False,
    ) -> None:
        compress = len(body) >= COMPRESS_MINIMUM_BYTES and "gzip" in self.headers.get(
            "Accept-Encoding", ""
        )
        if compress:
            body = gzip.compress(body, compresslevel=1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        if compress:
            self.send_header("Content-Encoding", "gzip")
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)


def new_token() -> str:
    """Return a random access token."""
    return secrets.token_urlsafe(24)


def make_server(
    catalogs: Catalogs,
    *,
    host: str,
    port: int,
    token: str,
    restart: Callable[[], None] | None = None,
    cache_bytes: int = 8 * 1024**3,
) -> VizServer:
    """Bind a server without starting it; port 0 selects a free port."""
    return VizServer((host, port), catalogs, token, restart=restart, cache_bytes=cache_bytes)
