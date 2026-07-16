#!/usr/bin/env bash
# Single entry point: build the self-contained analysis dashboard, serve it, and
# rebuild it every 60s so it live-updates as the analysis harness populates
# analysis-outputs/. Open the printed URL in a browser.
set -euo pipefail
cd "$(dirname "$0")/../.."
PORT="${DASH_PORT:-8420}"
DIR="analysis-outputs"

echo "[dashboard] initial build…"
python3 scripts/dashboard/build_dashboard.py

# serve the analysis-outputs dir (dashboard.html + the result tree)
python3 -m http.server "$PORT" --directory "$DIR" >/tmp/gge-dashboard-http.log 2>&1 &
HTTP_PID=$!
trap 'kill $HTTP_PID 2>/dev/null || true' EXIT

URL="http://localhost:${PORT}/dashboard.html"
echo "======================================================================"
echo "  ANALYST DASHBOARD LIVE  ->  $URL"
echo "  (auto-rebuilds every 60s; page auto-refreshes; Ctrl-C to stop)"
echo "======================================================================"

while true; do
  sleep 60
  python3 scripts/dashboard/build_dashboard.py || echo "[dashboard] rebuild error (kept last good html)"
done
