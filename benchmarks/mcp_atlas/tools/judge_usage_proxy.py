"""Localhost pass-through to an OpenAI-compatible judge endpoint that logs each response's usage.

usage: judge_usage_proxy.py PORT USAGE_JSONL
Reads JUDGE_BASE_URL (e.g. https://host/v1) and JUDGE_API_KEY from the environment; the key is
never logged.
"""

import http.server
import json
import os
import sys
import time
import urllib.request

KEY, BASE = os.environ["JUDGE_API_KEY"], os.environ["JUDGE_BASE_URL"].rstrip("/")
LOG = sys.argv[2]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        path = self.path[3:] if self.path.startswith("/v1") else self.path
        request = urllib.request.Request(
            BASE + path, data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                status, data = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, data = error.code, error.read()
        try:
            usage = json.loads(data).get("usage") or {}
            with open(LOG, "a") as handle:
                handle.write(json.dumps({"t": time.time(), "status": status, **usage}) + "\n")
        except (ValueError, AttributeError):
            pass
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
