"""
Attention Tracker — FastAPI Server
====================================
Serves the web app and optionally receives session logs.

Usage:
  python server.py                     # default port 8000
  python server.py --port 8080
  python server.py --host 0.0.0.0     # accessible on LAN (mobile devices)

For production TLS (HTTPS) put this behind nginx or Caddy.
"""

import argparse
import json
import socket
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WEB_DIR  = BASE_DIR / "web"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AttentionAI", version="1.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve static files (js, css) from web/
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/style.css", include_in_schema=False)
async def style():
    return FileResponse(WEB_DIR / "style.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def js():
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.post("/api/log", status_code=status.HTTP_200_OK)
async def save_log(request: Request):
    """
    Accept a JSON session log from the browser and save it as CSV.
    Body: { rows: [ { t, score, pitch, yaw, roll, earL, earR, gazeL, gazeR, status } ] }
    """
    try:
        body = await request.json()
        rows = body.get("rows", [])
        if not rows:
            return JSONResponse({"ok": False, "error": "no rows"})

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = LOGS_DIR / f"session_{ts}.csv"
        headers = ["t", "score", "pitch", "yaw", "roll",
                   "earL", "earR", "gazeL", "gazeR", "status"]
        lines = [",".join(headers)]
        for r in rows:
            lines.append(",".join(str(r.get(h, "")) for h in headers))

        out.write_text("\n".join(lines), encoding="utf-8")
        return JSONResponse({"ok": True, "saved": str(out)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)},
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Entry point ───────────────────────────────────────────────────────────────

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    parser = argparse.ArgumentParser(description="AttentionAI Web Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ip = get_local_ip()
    print("\n" + "=" * 52)
    print("  AttentionAI — Web Server")
    print("=" * 52)
    print(f"  Local   → http://localhost:{args.port}")
    print(f"  Network → http://{ip}:{args.port}  (mobile devices)")
    print("=" * 52)
    print("  Open the Network URL on your phone (same Wi-Fi)\n")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        log_level="warning",
        reload=False,
    )


if __name__ == "__main__":
    main()
