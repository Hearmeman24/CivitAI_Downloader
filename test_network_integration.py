#!/usr/bin/env python3
"""Local HTTP integration tests for the configured retrying session."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import download_with_aria as dwa


class RetryHandler(BaseHTTPRequestHandler):
    attempts = 0

    def do_GET(self):
        type(self).attempts += 1
        if self.attempts == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        if self.attempts == 2:
            self.send_response(503)
            self.end_headers()
            return
        payload = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class RetryingSessionIntegrationTests(unittest.TestCase):
    def test_session_retries_429_and_503_before_success(self):
        RetryHandler.attempts = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), RetryHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = dwa.build_http_session().get(
                f"http://127.0.0.1:{server.server_port}/metadata", timeout=2
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"ok": True})
            self.assertEqual(RetryHandler.attempts, 3)
            response.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
