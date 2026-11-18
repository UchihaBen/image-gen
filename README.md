# image-gen — API sinh ảnh Z-Image-Turbo trên GPU riêng

Sinh ảnh chất lượng cao bằng **Z-Image-Turbo** (chạy qua ComfyUI) trên server có GPU NVIDIA
(đã nhắm cho RTX 4070 12 GB), mở một cổng HTTP có API key để gọi từ máy khác.

```
Máy bạn ──HTTP + API key──▶ api_server.py (0.0.0.0:8000) ──▶ ComfyUI (127.0.0.1:8188, chỉ nội bộ) ──▶ GPU
```

## Yêu cầu trên server

- Linux, GPU NVIDIA, đã cài driver (`nvidia-smi` chạy được)
- `git`, `curl`
- Khoảng **25 GB đĩa trống**: model ~19.3 GB + PyTorch/ComfyUI ~5 GB
- RAM 16 GB trở lên (32 GB thì thoải mái)

## Cài và chạy (làm một lần)

```bash
git clone <URL repo GitHub của bạn> image-gen
cd image-gen

nvidia-smi                  # kiểm tra GPU và "CUDA Version" ở góc phải
cp .env.example .env
nano .env                   # đổi API_KEY (tạo bằng: openssl rand -hex 24)

bash setup.sh               # cài ComfyUI + PyTorch, ~5-10 phút
bash download_models.sh     # tải ~19 GB model, chạy lại được nếu đứt mạng
bash start.sh               # chạy server
```

Nếu `nvidia-smi` báo **CUDA Version thấp hơn 12.8**, bỏ dấu `#` ở dòng `TORCH_INDEX` trong `.env`
(dùng cu126) trước khi chạy `setup.sh`.

### Chạy nền (tắt SSH vẫn chạy)

```bash
tmux new -s imagegen        # mở phiên tmux
bash start.sh               # chạy trong tmux, rồi bấm Ctrl+B rồi D để thoát ra
tmux attach -t imagegen     # quay lại xem log
```

Hoặc: `nohup bash start.sh > logs/server.log 2>&1 &`

### Mở cổng ra ngoài

```bash
sudo ufw allow 8000/tcp     # nếu server dùng ufw
```

Nếu server là VPS/cloud, mở thêm cổng 8000 TCP trong security group / firewall của nhà cung cấp.
**Không** mở cổng 8188 (ComfyUI không có mật khẩu).

## Gọi API

Kiểm tra (không cần key):

```bash
curl http://SERVER:8000/health
```

Sinh ảnh, nhận file PNG:

```bash
curl -X POST http://SERVER:8000/generate \
  -H "X-API-Key: KEY_CUA_BAN" -H "Content-Type: application/json" \
  -d '{"prompt": "cinematic shot of a lone samurai in the rain, neon lights, 35mm film", "width": 1344, "height": 768}' \
  -o anh.png
```

Tham số của `POST /generate` (JSON):

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `prompt` | bắt buộc | Viết tiếng Anh cho kết quả tốt nhất |
| `width`, `height` | 1024 | Bội số của 16, từ 256 đến 2048. Video 16:9 nên dùng **1344×768**, dọc 9:16 dùng **768×1344** |
| `seed` | -1 | -1 là ngẫu nhiên. Giữ cùng seed để ra ảnh giống nhau |
| `steps` | 8 | Model turbo, 8 là chuẩn |
| `cfg` | 1.0 | Giữ nguyên |
| `shift` | 3.0 | Giữ nguyên |
| `as_base64` | false | `true` thì trả JSON `{seed, elapsed_s, image_base64}` thay vì file PNG |

Header trả về: `X-Seed` (seed đã dùng), `X-Elapsed-S` (số giây). Tài liệu tương tác: `http://SERVER:8000/docs`.

Gọi bằng Python:

```python
import requests
r = requests.post("http://SERVER:8000/generate",
                  headers={"X-API-Key": "KEY_CUA_BAN"},
                  json={"prompt": "a red fox in snow, golden hour", "width": 1344, "height": 768})
r.raise_for_status()
open("fox.png", "wb").write(r.content)
```

Sinh hàng loạt cho video (mỗi dòng trong `prompts.txt` là một cảnh, ảnh lưu thành `0001.png`, `0002.png`...):

```bash
python client_example.py --url http://SERVER:8000 --key KEY_CUA_BAN --prompts prompts.txt --out frames
```

## Lưu ý

- **Lần gọi đầu tiên chậm** (30-90 giây) vì phải nạp model vào GPU. Các lần sau khoảng vài giây mỗi ảnh.
- Request được xếp hàng và xử lý lần lượt trên GPU.
- Thiếu VRAM/RAM: đặt `TEXT_ENCODER_NAME=qwen_3_4b_fp8_mixed.safetensors` trong `.env`, chạy lại
  `bash download_models.sh`, hoặc thêm `COMFY_ARGS=--lowvram`.
- Log ComfyUI ở `logs/comfyui.log`.
- Cập nhật: `git pull && bash setup.sh`, rồi chạy lại `start.sh`.
- Giấy phép model: Z-Image-Turbo phát hành theo Apache 2.0.
