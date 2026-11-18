#!/usr/bin/env bash
# Install ComfyUI + PyTorch (CUDA) + API server deps into ./.venv
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

command -v git >/dev/null || { echo "Cần cài git trước"; exit 1; }
command -v nvidia-smi >/dev/null || echo "CẢNH BÁO: không thấy nvidia-smi, kiểm tra driver NVIDIA."

if ! command -v uv >/dev/null; then
  echo ">> Cài uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ComfyUI ]; then
  echo ">> Clone ComfyUI"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
else
  echo ">> Cập nhật ComfyUI"
  git -C ComfyUI pull --ff-only
fi

[ -d .venv ] || uv venv --python 3.12 .venv

echo ">> Cài PyTorch từ $TORCH_INDEX"
uv pip install --python .venv torch torchvision torchaudio --index-url "$TORCH_INDEX"
echo ">> Cài thư viện ComfyUI và API server"
uv pip install --python .venv -r ComfyUI/requirements.txt
uv pip install --python .venv -r requirements.txt

.venv/bin/python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print(f">> torch {torch.__version__}, CUDA: {ok}" + (f", GPU: {torch.cuda.get_device_name(0)}" if ok else ""))
if not ok:
    raise SystemExit("PyTorch không thấy GPU. Thử TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash setup.sh")
EOF
echo ">> Xong. Bước tiếp: bash download_models.sh"
