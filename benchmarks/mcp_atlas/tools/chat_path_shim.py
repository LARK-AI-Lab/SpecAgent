"""Map the harness's POST /v1/chat/completions onto a provider whose chat path differs.

usage: chat_path_shim.py PORT
The harness appends /v1/chat/completions to LLM_BASE_URL; some providers (e.g. Zhipu, whose
path is /api/paas/v4/chat/completions) do not serve that path. Reads UPSTREAM_CHAT_URL (the
full chat-completions URL) and UPSTREAM_API_KEY from the environment; honours HTTPS_PROXY.
"""

import http.server
import json
import os
import sys
import urllib.request

URL, KEY = os.environ["UPSTREAM_CHAT_URL"], os.environ["UPSTREAM_API_KEY"]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler())


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        request = urllib.request.Request(
            URL, data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
        )
        try:
            response = OPENER.open(request, timeout=900)
            status, data = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, data = error.code, error.read()
        except Exception as error:  # noqa: BLE001 - surfaced to the harness as a 502
            status = 502
            data = json.dumps({"error": {"message": f"shim transport error: {error!r}"}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
