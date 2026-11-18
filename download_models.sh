#!/usr/bin/env bash
# Download Z-Image-Turbo weights into ComfyUI/models (resumable, skips finished files)
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a

[ -d ComfyUI ] || { echo "Chạy bash setup.sh trước"; exit 1; }

BASE="${HF_ENDPOINT:-https://huggingface.co}/Comfy-Org/z_image_turbo/resolve/main/split_files"
UNET="${UNET_NAME:-z_image_turbo_bf16.safetensors}"
TE="${TEXT_ENCODER_NAME:-qwen_3_4b.safetensors}"
VAE="${VAE_NAME:-ae.safetensors}"

dl() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then echo ">> Đã có $out"; return; fi
  echo ">> Tải $out"
  curl -L --fail --retry 5 -C - -o "$out.part" "$url"
  mv "$out.part" "$out"
}

dl "$BASE/diffusion_models/$UNET" "ComfyUI/models/diffusion_models/$UNET"
dl "$BASE/text_encoders/$TE"      "ComfyUI/models/text_encoders/$TE"
dl "$BASE/vae/$VAE"               "ComfyUI/models/vae/$VAE"
echo ">> Xong. Bước tiếp: bash start.sh"
