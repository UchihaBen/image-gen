# image-gen — API sinh ảnh trên GPU riêng

API HTTP sinh ảnh chạy trên server có GPU NVIDIA (đã nhắm cho RTX 4070 12 GB), qua ComfyUI:

| Chức năng | Endpoint | Model |
|---|---|---|
| Tạo ảnh từ prompt | `POST /generate` | Z-Image-Turbo |
| Biến thể từ ảnh mẫu | `POST /img2img` | Z-Image-Turbo |
| Sửa ảnh bằng lời, giữ nhân vật | `POST /edit` | FLUX.1 Kontext dev |
| Giữ bố cục / dáng / đường nét | `POST /control` | Z-Image-Turbo + Fun ControlNet Union |

Có **trang thử** tại `/` và tài liệu tương tác tại `/docs`. Ảnh trả về ngay, không lưu trên server.

```
Máy bạn ──HTTP + API key──▶ api_server.py ──▶ ComfyUI (127.0.0.1:8188, chỉ nội bộ) ──▶ GPU
```

## Cài đặt (làm một lần trên server)

Yêu cầu: Linux, GPU NVIDIA đã có driver (`nvidia-smi` chạy được), `git`, `curl`, RAM 32 GB trở lên.

```bash
git clone https://github.com/UchihaBen/image-gen.git && cd image-gen
cp .env.example .env
sed -i "s/^API_KEY=.*/API_KEY=$(openssl rand -hex 24)/" .env   # tạo API key
nano .env                                                       # sửa PORT/HOST nếu cần

bash setup.sh                       # ComfyUI + PyTorch, ~5-10 phút
bash download_models.sh             # cơ bản: /generate, /img2img   (~19.3 GB)
bash download_models.sh control     # thêm /control                  (+2.9 GB)
bash download_models.sh edit        # thêm /edit                     (+16.1 GB)
```

Chạy nền bằng tmux:

```bash
tmux new -s imagegen
bash start.sh          # đợi log "Warm-up done", rồi Ctrl+B, D để thoát
```

Cập nhật code: `git pull && bash setup.sh`, rồi khởi động lại `start.sh` trong tmux.

Nếu đặt sau Cloudflare Tunnel hoặc nginx trên cùng máy, dùng `HOST=127.0.0.1` để không lộ cổng ra ngoài.
Không bao giờ mở cổng ComfyUI (8188) ra ngoài.

## Xác thực

Mọi endpoint trừ `/`, `/health`, `/docs` cần header `X-API-Key: <key>` (hoặc `Authorization: Bearer <key>`).

## Chế độ chờ và chế độ job

Mặc định (`wait=true`) request giữ kết nối và trả ảnh PNG khi xong.

Với `wait=false`, server trả ngay `202 {"job_id": ...}`. Gọi `GET /jobs/{job_id}`:
`202` là đang chạy, `200` là ảnh PNG (job bị xoá sau khi lấy), lỗi thì trả JSON `detail`.
Kết quả chưa lấy tự xoá sau 15 phút (`JOB_TTL`).

**Đứng sau Cloudflare**, request quá 100 giây bị cắt (lỗi 524). `/edit` thường mất 30-90 giây, nên dùng `wait=false`.

## Các endpoint

### `POST /generate` (JSON)

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `prompt` | bắt buộc | Tiếng Anh cho kết quả tốt nhất |
| `width`, `height` | 1024 | 256-2048, bội số của 16. Video: 1344×768 hoặc 768×1344 |
| `steps` | 8 | |
| `seed` | -1 | -1 = ngẫu nhiên |
| `cfg`, `shift` | 1.0, 3.0 | Giữ nguyên |
| `as_base64`, `wait` | false, true | |

```bash
curl -X POST $URL/generate -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"a red fox in snow, golden hour","width":1344,"height":768}' -o fox.png
```

### `POST /img2img` (multipart form)

Giữ bố cục và màu của ảnh mẫu, vẽ lại theo prompt.

| Trường | Mặc định | Ghi chú |
|---|---|---|
| `image` | bắt buộc | PNG/JPEG/WebP, tối đa 20 MB |
| `prompt` | bắt buộc | Mô tả ảnh muốn ra |
| `strength` | 0.6 | 0.05-1. Thấp: gần giống ảnh mẫu. Cao: vẽ lại nhiều |
| `megapixels` | 1.0 | 0.25-2. Kích thước ảnh ra, giữ tỉ lệ ảnh mẫu |
| `steps`, `seed`, `as_base64`, `wait` | 8, -1, false, true | |

```bash
curl -X POST $URL/img2img -H "X-API-Key: $KEY" \
  -F image=@photo.jpg -F prompt="oil painting, impressionist style" -F strength=0.55 -o out.png
```

### `POST /edit` (multipart form)

Sửa ảnh bằng câu lệnh, giữ nguyên nhân vật hoặc vật thể.

| Trường | Mặc định | Ghi chú |
|---|---|---|
| `image` | bắt buộc | Ảnh cần sửa |
| `prompt` | bắt buộc | Câu lệnh tiếng Anh, ví dụ `change the background to a beach at sunset` |
| `image2` | không | Ảnh tham chiếu thứ hai, ghép bên phải ảnh 1 (ví dụ nhân vật cần đưa vào cảnh) |
| `guidance` | 2.5 | 1-10 |
| `steps` | 20 | |
| `megapixels`, `seed`, `as_base64`, `wait` | 1.0, -1, false, true | |

```bash
curl -X POST $URL/edit -H "X-API-Key: $KEY" -F image=@girl.jpg -F wait=false \
  -F prompt="make her wear a red raincoat, keep her face exactly the same"
# => {"job_id":"abc...","status":"running",...}
curl $URL/jobs/abc... -H "X-API-Key: $KEY" -o edited.png   # gọi lại tới khi nhận được ảnh
```

### `POST /control` (multipart form)

Giữ đường nét, dáng người hoặc bố cục của ảnh mẫu, nội dung vẽ mới theo prompt.

| Trường | Mặc định | Ghi chú |
|---|---|---|
| `image` | bắt buộc | Ảnh mẫu |
| `prompt` | bắt buộc | Nội dung mới |
| `mode` | `canny` | `canny`: tự tách đường nét từ ảnh thường. `direct`: ảnh gửi lên đã là ảnh điều khiển (pose, depth, lineart) |
| `strength` | 0.9 | 0-2. Độ bám bố cục |
| `canny_low`, `canny_high` | 0.1, 0.32 | Ngưỡng tách nét |
| `megapixels`, `steps`, `seed`, `as_base64`, `wait` | 1.0, 8, -1, false, true | |

```bash
curl -X POST $URL/control -H "X-API-Key: $KEY" \
  -F image=@room.jpg -F prompt="futuristic sci-fi control room, neon lights" -o scifi.png
```

### `GET /jobs/{job_id}`, `GET /health`

`/health` (không cần key) cho biết GPU, VRAM và chức năng nào đã cài (`features`).

## Phản hồi và lỗi

Ảnh trả về `image/png` kèm header `X-Seed` và `X-Elapsed-S`. Với `as_base64=true`:
`{"seed": ..., "elapsed_s": ..., "image_base64": "..."}`.

| Mã | Ý nghĩa |
|---|---|
| 401 | Sai hoặc thiếu API key |
| 400 | Ảnh không đọc được, kích thước không chia hết cho 16, ngưỡng canny sai |
| 413 | Ảnh lớn hơn 20 MB |
| 422 | Tham số sai kiểu hoặc ngoài giới hạn |
| 503 | ComfyUI chưa chạy, hoặc chức năng chưa tải model (`detail` ghi lệnh cần chạy) |
| 500 / 504 | Sinh ảnh lỗi / quá thời gian |

## Lưu ý

- Một GPU xử lý lần lượt: request gửi cùng lúc sẽ xếp hàng.
- Đổi chức năng (ví dụ từ `/generate` sang `/edit`) lần đầu chậm thêm vài giây để đổi model trong GPU.
- Sinh hàng loạt cho video: `python client_example.py --url $URL --key $KEY --prompts prompts.txt --out frames`.
- Thiếu VRAM/RAM: `TEXT_ENCODER_NAME=qwen_3_4b_fp8_mixed.safetensors` hoặc `COMFY_ARGS=--lowvram` trong `.env`.
- Log ComfyUI: `logs/comfyui.log`.
- Giấy phép: Z-Image-Turbo và ControlNet là Apache 2.0. FLUX.1 Kontext dev chỉ dùng phi thương mại.
