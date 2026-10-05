#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT_DIR"
PROJECT_PYTHON="$ROOT_DIR/venv/bin/python"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -x "$PROJECT_PYTHON" ]] || fail "Backend venv is missing. Run bash setup-scripts/setup-new-machine.sh first."
[[ -d "$ROOT_DIR/frontend/node_modules" ]] || fail "Frontend packages are missing. Run bash setup-scripts/setup-new-machine.sh first."
command -v screen >/dev/null 2>&1 || fail "GNU screen is not installed."
command -v ngrok >/dev/null 2>&1 || fail "ngrok is not installed or is not on PATH. Install and authenticate ngrok first."
command -v npm >/dev/null 2>&1 || fail "npm is not installed or is not on PATH."

# The tunnel exposes the app publicly; do not start it with the shared dev secret.
"$PROJECT_PYTHON" - <<'PY'
from app.core.config.settings import get_settings

secret = get_settings().auth_jwt_secret.strip()
if len(secret) < 32 or secret in {
    "change-this-development-secret-before-deployment",
    "replace-with-a-long-random-secret-before-deployment",
}:
    raise SystemExit(
        "ERROR: Set a unique AUTH_JWT_SECRET (at least 32 characters) in the root .env "
        "before opening an ngrok tunnel."
    )
PY

existing_screens="$(screen -ls 2>&1 || true)"
if [[ "$existing_screens" =~ \.(fontend|backend|ngrok)[[:space:]] ]]; then
    fail "A screen session named fontend, backend, or ngrok already exists. Check screen -ls; this script will not replace it."
fi

"$PROJECT_PYTHON" - <<'PY'
import socket

for port in (8000, 5173, 4040):
    with socket.socket() as connection:
        connection.settimeout(0.5)
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"ERROR: Port {port} is already in use. Stop the existing service before starting new screens.")
PY

started_screens=()
cleanup_if_failed() {
    if (( $? != 0 )); then
        echo "Startup failed. Stopping only the screen sessions created by this script." >&2
        for session in "${started_screens[@]}"; do
            screen -S "$session" -X quit >/dev/null 2>&1 || true
        done
        echo "Check backend.log, fontend.log, or ngrok.log in $ROOT_DIR for details." >&2
    fi
}
trap cleanup_if_failed EXIT

wait_for_http() {
    "$PROJECT_PYTHON" - "$1" "${2:-}" <<'PY'
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

url = sys.argv[1]
host = sys.argv[2]
opener = build_opener(ProxyHandler({}))
deadline = time.monotonic() + 25
while time.monotonic() < deadline:
    try:
        request = Request(url, headers={"Host": host} if host else {})
        with opener.open(request, timeout=2) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(0.5)
raise SystemExit(f"ERROR: {url} did not become ready within 25 seconds")
PY
}

echo "Starting Backend..."
screen -dmS backend bash -c \
    'cd "$1" && exec "$2" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 >> "$3" 2>&1' \
    _ "$ROOT_DIR" "$PROJECT_PYTHON" "$ROOT_DIR/backend.log"
started_screens+=(backend)
wait_for_http "http://127.0.0.1:8000/openapi.json"

echo "Starting ngrok..."
NGROK_BIN="$(command -v ngrok)"
screen -dmS ngrok bash -c 'exec "$1" http 5173 >> "$2" 2>&1' \
    _ "$NGROK_BIN" "$ROOT_DIR/ngrok.log"
started_screens+=(ngrok)

public_url="$("$PROJECT_PYTHON" - <<'PY'
import json
import time
from urllib.request import ProxyHandler, build_opener

opener = build_opener(ProxyHandler({}))
deadline = time.monotonic() + 25
while time.monotonic() < deadline:
    try:
        with opener.open("http://127.0.0.1:4040/api/tunnels", timeout=2) as response:
            tunnels = json.load(response).get("tunnels", [])
        for tunnel in tunnels:
            url = tunnel.get("public_url", "")
            if url.startswith("https://"):
                print(url)
                raise SystemExit(0)
    except Exception:
        pass
    time.sleep(0.5)
raise SystemExit("ERROR: ngrok did not provide an HTTPS URL within 25 seconds")
PY
)"
public_host="$("$PROJECT_PYTHON" - "$public_url" <<'PY'
import sys
from urllib.parse import urlsplit

host = urlsplit(sys.argv[1]).hostname
if not host:
    raise SystemExit("ERROR: ngrok returned an invalid public URL")
print(host)
PY
)"

echo "Starting Frontend for $public_url..."
# The ngrok hostname is allowed only for this Vite process; no .env changes.
# Force same-origin /api so public browsers never call their own localhost.
screen -dmS fontend bash -c \
    'cd "$1/frontend" && exec env __VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS="$2" VITE_API_BASE_URL=/api API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev >> "$3" 2>&1' \
    _ "$ROOT_DIR" "$public_host" "$ROOT_DIR/fontend.log"
started_screens+=(fontend)
wait_for_http "http://127.0.0.1:5173/api/openapi.json" "$public_host"

trap - EXIT
echo
echo "Ready: $public_url"
echo "Local: http://127.0.0.1:5173"
echo "Logs: backend.log, fontend.log, ngrok.log in $ROOT_DIR"
echo "Attach: screen -r backend | screen -r fontend | screen -r ngrok"
echo "Stop: screen -S backend -X quit; screen -S fontend -X quit; screen -S ngrok -X quit"
screen -ls
