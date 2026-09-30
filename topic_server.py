# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Optional localhost-only editor for the otherwise static usage dashboard."""
import hmac
import json
import secrets
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from topic_classifier import apply_edit, catalog_lock, load_catalog, save_catalog


def make_server(html_path, topic_path):
    html_path, topic_path = Path(html_path), Path(topic_path)
    if not html_path.is_file():
        raise FileNotFoundError(html_path)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(content)

        def _json(self, status, value):
            self._send(status, json.dumps(value).encode("utf-8"), "application/json; charset=utf-8")

        def _valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def _authorized(self):
            received = self.headers.get("X-Topic-Token", "")
            return hmac.compare_digest(received, token)

        def do_GET(self):
            if not self._valid_host():
                self._json(403, {"error": "invalid host"})
            elif self.path == "/":
                try:
                    html = html_path.read_text(encoding="utf-8")
                except OSError as exc:
                    self._json(500, {"error": f"could not read dashboard: {exc}"})
                    return
                marker = "const TOPIC_EDIT_TOKEN = null;"
                if marker not in html:
                    self._json(500, {"error": "dashboard has no topic editor"})
                    return
                html = html.replace(marker, f"const TOPIC_EDIT_TOKEN = {json.dumps(token)};", 1)
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/topics" and self._authorized():
                try:
                    self._json(200, load_catalog(topic_path))
                except ValueError as exc:
                    self._json(500, {"error": str(exc)})
            else:
                self._json(403 if self.path == "/api/topics" else 404, {"error": "not found"})

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if not self._valid_host() or self.headers.get("Origin") != origin or not self._authorized():
                self._json(403, {"error": "unauthorized origin or token"})
                return
            if self.path != "/api/topics":
                self._json(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 16384 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("invalid topic edit request")
                edit = json.loads(self.rfile.read(size))
                if not isinstance(edit, dict):
                    raise ValueError("topic edit must be an object")
                with catalog_lock(topic_path):
                    catalog = load_catalog(topic_path)
                    apply_edit(catalog, edit)
                    save_catalog(topic_path, catalog)
            except (ValueError, UnicodeDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            except (OSError, sqlite3.Error) as exc:
                self._json(500, {"error": f"could not save topic file: {exc}"})
                return
            self._json(200, catalog)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.topic_token = token
    return server
