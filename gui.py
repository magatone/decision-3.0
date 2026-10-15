# Minimal browser GUI for d3 models: stdlib HTTP server + model in-process.
# Run: .venv\Scripts\python.exe gui.py [--model vllm-sr/d3-mini] [--port 7860]  ->  http://127.0.0.1:7860
# GUI_AUTH=user:pass env turns on browser Basic Auth (set it before exposing beyond localhost, e.g. Docker/tunnel).
# ponytail: one request at a time (global lock), same as d3_server.py; use Engine mode for concurrency.
import argparse
import base64
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModel

PINNED = {"vllm-sr/d3-mini": "6d8971f464d6f9d4c5af7d9a8d75f3aa03be4f59"}
HTML = Path(__file__).with_name("gui.html")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vllm-sr/d3-mini")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--host", default="127.0.0.1")  # Docker needs 0.0.0.0
args = ap.parse_args()
AUTH = os.environ.get("GUI_AUTH", "")
EXPECTED = "Basic " + base64.b64encode(AUTH.encode()).decode() if AUTH else ""
MAX_BODY = 64 * 1024 * 1024  # ~30 images at 1.6MP as base64; stops one request from eating RAM

print(f"loading {args.model} ...")
model = AutoModel.from_pretrained(args.model, revision=PINNED.get(args.model), trust_remote_code=True)
lock = threading.Lock()
# First call allocates/compiles (~1s); do it now so the first click shows real latency.
model.system_one(state="warm up", questions={"q": {"type": "noul", "instructions": "Is this a test?"}})
print(f"ready: http://{args.host}:{args.port}" + ("" if AUTH else "  (no auth)"))


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authed(self):
        if not EXPECTED or hmac.compare_digest(self.headers.get("Authorization", ""), EXPECTED):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="d3"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_GET(self):
        if not self.authed():
            return
        if self.path in ("/", "/index.html"):
            self.send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.authed():
            return
        if self.path != "/api/run":
            return self.send(404, {"error": "not found"})
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            self.close_connection = True  # body left unread
            return self.send(413, {"error": f"request too large ({size // 2**20}MB > {MAX_BODY // 2**20}MB)"})
        try:
            req = json.loads(self.rfile.read(size))
            with lock:
                t = time.perf_counter()
                result = model.system_one(
                    state=req.get("state") or "",
                    questions=req["questions"],
                    images=req.get("images") or None,
                    videos=req.get("videos") or None,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t) * 1000
            self.send(200, {"model": args.model, "ms": round(ms, 1), "answers": result["answers"]})
        except Exception as e:  # show model-side validation errors in the page instead of a dead request
            self.send(400, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *a):
        pass


# Default 127.0.0.1. With --host 0.0.0.0 set GUI_AUTH or put an auth layer (Cloudflare Access) in front.
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
