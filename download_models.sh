#!/usr/bin/env bash
# Download model weights into ComfyUI/models (resumable, skips finished files).
#
#   bash download_models.sh                 # cơ bản: /generate, /img2img      (~19.3 GB)
#   bash download_models.sh control         # + /control  (ControlNet)         (+2.9 GB)
#   bash download_models.sh edit            # + /edit     (FLUX.1 Kontext dev) (+16.1 GB)
#   bash download_models.sh all             # tất cả
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && . ./.env; set +a

[ -d ComfyUI ] || { echo "Chạy bash setup.sh trước"; exit 1; }

HF="${HF_ENDPOINT:-https://huggingface.co}"
WANT=" ${*:-base} "
want() { [[ "$WANT" == *" $1 "* || "$WANT" == *" all "* ]]; }

dl() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then echo ">> Đã có $out"; return; fi
  echo ">> Tải $out"
  curl -L --fail --retry 5 -C - -o "$out.part" "$url"
  mv "$out.part" "$out"
}

echo ">> Dung lượng trống: $(df -h . | awk 'NR==2 {print $4}')"
M=ComfyUI/models

# Cơ bản (luôn cần): Z-Image-Turbo
Z="$HF/Comfy-Org/z_image_turbo/resolve/main/split_files"
dl "$Z/diffusion_models/${UNET_NAME:-z_image_turbo_bf16.safetensors}" "$M/diffusion_models/${UNET_NAME:-z_image_turbo_bf16.safetensors}"
dl "$Z/text_encoders/${TEXT_ENCODER_NAME:-qwen_3_4b.safetensors}"     "$M/text_encoders/${TEXT_ENCODER_NAME:-qwen_3_4b.safetensors}"
dl "$Z/vae/${VAE_NAME:-ae.safetensors}"                                "$M/vae/${VAE_NAME:-ae.safetensors}"

if want control; then
  dl "$HF/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors" \
     "$M/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors"
fi

if want edit; then
  dl "$HF/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors" \
     "$M/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors"
  dl "$HF/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"               "$M/text_encoders/clip_l.safetensors"
  dl "$HF/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors" "$M/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"
fi

echo ">> Xong. Còn trống: $(df -h . | awk 'NR==2 {print $4}'). Khởi động lại start.sh để dùng tính năng mới."
