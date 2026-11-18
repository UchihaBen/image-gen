#!/usr/bin/env bash
# Start ComfyUI (localhost only) + the API server.
# If either process dies, the other is stopped and the script exits non-zero, so a
# supervisor (systemd, see image-gen.service) can restart the whole service.
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
echo ">> $(date '+%F %T') Khởi động ComfyUI trên 127.0.0.1:$COMFY_PORT (log: logs/comfyui.log)"
.venv/bin/python ComfyUI/main.py --listen 127.0.0.1 --port "$COMFY_PORT" \
  --disable-auto-launch ${COMFY_ARGS:-} >> logs/comfyui.log 2>&1 &
COMFY_PID=$!
API_PID=""
cleanup() { kill $COMFY_PID ${API_PID:-} 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null && break
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "ComfyUI bị lỗi khi khởi động:"; tail -n 40 logs/comfyui.log; exit 1
  fi
  sleep 1
done

export COMFY_URL="http://127.0.0.1:$COMFY_PORT" HOST PORT
echo ">> API server: http://$HOST:$PORT  (trang thử: /, tài liệu: /docs)"
.venv/bin/python api_server.py &
API_PID=$!

set +e
wait -n "$COMFY_PID" "$API_PID"
code=$?
set -e
if ! kill -0 "$COMFY_PID" 2>/dev/null; then
  echo ">> $(date '+%F %T') ComfyUI đã dừng (mã $code). 40 dòng log cuối:"; tail -n 40 logs/comfyui.log
else
  echo ">> $(date '+%F %T') API server đã dừng (mã $code)."
fi
exit 1
