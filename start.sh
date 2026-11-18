#!/usr/bin/env bash
# Start ComfyUI (localhost only) + the public API server. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
COMFY_PORT="${COMFY_PORT:-8188}"

if [ -z "${API_KEY:-}" ] && [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
  echo "Chưa đặt API_KEY trong .env mà lại mở cổng public. Dừng."; exit 1
fi

mkdir -p logs
echo ">> Khởi động ComfyUI trên 127.0.0.1:$COMFY_PORT (log: logs/comfyui.log)"
.venv/bin/python ComfyUI/main.py --listen 127.0.0.1 --port "$COMFY_PORT" \
  --disable-auto-launch ${COMFY_ARGS:-} > logs/comfyui.log 2>&1 &
COMFY_PID=$!
trap 'kill $COMFY_PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null && break
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "ComfyUI bị lỗi khi khởi động:"; tail -n 40 logs/comfyui.log; exit 1
  fi
  sleep 1
done

export COMFY_URL="http://127.0.0.1:$COMFY_PORT" HOST PORT
echo ">> API server: http://$HOST:$PORT  (tài liệu: /docs)"
.venv/bin/python api_server.py
