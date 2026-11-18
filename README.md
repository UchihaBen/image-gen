# image-gen — API sinh ảnh trên GPU riêng

API HTTP sinh ảnh chạy trên server có GPU NVIDIA (đã nhắm cho RTX 4070 12 GB), qua ComfyUI:

| Chức năng | Endpoint | Model |
|---|---|---|
| Tạo ảnh từ prompt | `POST /generate` | Z-Image-Turbo |
| Biến thể từ ảnh mẫu | `POST /img2img` | Z-Image-Turbo |
| Giữ bố cục / dáng / đường nét | `POST /control` | Z-Image-Turbo + Fun ControlNet Union |
| Sửa ảnh bằng lời, giữ nhân vật | `POST /edit` | FLUX.1 Kontext dev |

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

### Chạy như dịch vụ (khuyên dùng: tự khởi động lại khi lỗi hoặc khi máy khởi động)

```bash
sudo cp image-gen.service /etc/systemd/system/      # sửa User/đường dẫn trong file nếu khác
sudo systemctl daemon-reload
sudo systemctl enable --now image-gen
journalctl -u image-gen -f                          # xem log
```

`start.sh` thoát ngay khi ComfyUI hoặc API server chết, và systemd khởi động lại cả hai sau 10 giây.
Chạy tay thì dùng `bash start.sh` trong tmux.

Cập nhật code: `git pull && .venv/bin/pip install -r requirements.txt` (hoặc `uv pip install --python .venv -r requirements.txt`),
rồi `sudo systemctl restart image-gen`.

Nếu đặt sau Cloudflare Tunnel hoặc nginx trên cùng máy, dùng `HOST=127.0.0.1`. Không mở cổng ComfyUI (8188) ra ngoài.

### LoRA phong cách

Chép file `.safetensors` vào `ComfyUI/models/loras/`. API chỉ dùng LoRA có trong thư mục này
(`GET /loras` liệt kê). Chọn LoRA làm cho **Z-Image-Turbo**; LoRA của model khác không dùng được.

## Xác thực

Mọi endpoint trừ `/`, `/health`, `/docs` cần header `X-API-Key: <key>` (hoặc `Authorization: Bearer <key>`).

## Chờ kết quả và job

Mọi request chạy như một job.

- `wait=true` (mặc định): request chờ tối đa `SYNC_WAIT` giây (85). Xong trong thời gian đó thì nhận ảnh luôn;
  chưa xong thì nhận `202` với `job_id`, `queue_position`, `result_url`.
- `wait=false`: nhận `202` ngay.
- `GET /jobs/{job_id}`: `202` khi đang chạy (kèm `queue_position`, `images_done`), `200` là kết quả
  (job bị xoá sau khi lấy), lỗi thì JSON `detail`. Kết quả chưa lấy tự xoá sau 15 phút.

Vì vậy client nên luôn xử lý được `202`. Cloudflare cắt request quá 100 giây, nên không bao giờ chờ lâu hơn.

## Tham số chung (`/generate`, `/img2img`, `/control`)

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `prompt` | bắt buộc | Tiếng Anh, phong cách đặt ở đầu |
| `n` | 1 | 1-4 ảnh. Seed lần lượt `seed`, `seed+1`, ... |
| `seed` | -1 | -1 là ngẫu nhiên (luôn < 2^53 để JavaScript đọc đúng) |
| `steps` | 8 | |
| `format` | `png` | `png`, `jpeg`, `webp`. 1920×1080: PNG 3-4 MB, WebP khoảng 0.3-0.5 MB |
| `quality` | 90 | 50-100, cho jpeg/webp |
| `lora`, `lora_strength` | không, 0.8 | Tên trong `GET /loras` |
| `cfg`, `negative_prompt` | 1.0, "" | Thử nghiệm: `negative_prompt` chỉ có tác dụng khi `cfg` > 1, chậm gấp đôi, có thể làm ảnh gắt màu |
| `shift` | 3.0 | |
| `as_base64` | false | |
| `wait` | true | Xem phần trên |

**Kết quả:** 1 ảnh và `as_base64=false` → file ảnh, header `X-Seed`, `X-Elapsed-S`, `X-Image-Width`, `X-Image-Height`.
Ngược lại (`n` > 1 hoặc `as_base64=true`) → JSON:

```json
{"elapsed_s": 41.2, "format": "webp", "count": 2,
 "images": [{"seed": 101, "width": 1920, "height": 1080, "format": "webp", "image_base64": "..."},
            {"seed": 102, "width": 1920, "height": 1080, "format": "webp", "image_base64": "..."}]}
```

### `POST /generate` (JSON)

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `width`, `height` | 1024 | 256-2048, **đúng từng pixel** (vd 1920×1080). Server gen ở bội số 16 gần nhất rồi cắt giữa |
| `hires` | false | Gen 2 lượt: nháp ở kích thước / `hires_scale`, phóng to, tinh chỉnh với `hires_denoise` |
| `hires_scale`, `hires_denoise` | 1.5, 0.35 | |

```bash
curl -X POST $URL/generate -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"digital painting, ancient Chinese palace at dusk","width":1920,"height":1080,"format":"webp"}' -o palace.webp
```

### `POST /img2img` (multipart form)

Giữ bố cục và màu của ảnh mẫu, vẽ lại theo prompt. Người hay vật không có trong prompt có thể bị biến thành thứ khác,
nên prompt cần mô tả đủ mọi chủ thể muốn giữ.

| Trường | Mặc định | Ghi chú |
|---|---|---|
| `image` | bắt buộc | PNG/JPEG/WebP, tối đa 30 MB |
| `strength` | 0.6 | 0.05-1. Thấp: gần giống ảnh mẫu. Cao: vẽ lại nhiều |
| `width`, `height` | 0 | Kích thước ra đúng từng pixel; ảnh mẫu được cắt giữa cho vừa tỉ lệ. 0 = theo `megapixels` |
| `megapixels` | 1.0 | 0.25-2, giữ tỉ lệ ảnh mẫu (cạnh làm tròn theo 16) |

### `POST /control` (multipart form)

Giữ đường nét, dáng người hoặc bố cục của ảnh mẫu, nội dung vẽ mới theo prompt (cũng cần mô tả đủ chủ thể).

| Trường | Mặc định | Ghi chú |
|---|---|---|
| `image` | bắt buộc | Ảnh mẫu |
| `mode` | `canny` | `canny`: tự tách nét từ ảnh thường. `direct`: ảnh đã là ảnh điều khiển (pose, depth, lineart) |
| `strength` | 0.9 | 0-2. Độ bám bố cục |
| `canny_low`, `canny_high` | 0.1, 0.32 | |
| `width`, `height`, `megapixels` | 0, 0, 1.0 | Như `/img2img` |

### `POST /edit` (multipart form)

Sửa ảnh bằng câu lệnh, giữ nguyên nhân vật. Trường: `image`, `prompt`, `image2` (tuỳ chọn), `guidance` (2.5),
`steps` (20), `megapixels`, `n`, `seed`, `format`, `quality`, `as_base64`, `wait`.

### `GET /loras`, `GET /health`

`/health` (không cần key): `backend` (`up`/`down`), `features`, `jobs_pending` (mọi request đang chạy),
`gpu_queue` (`running`, `waiting`), số LoRA, GPU/VRAM.

## Mã lỗi

| Mã | Ý nghĩa |
|---|---|
| 202 | Đang chạy, lấy kết quả ở `result_url` |
| 400 | Ảnh không đọc được, thiếu một trong `width`/`height`, LoRA không có, ngưỡng canny sai |
| 401 | Sai hoặc thiếu API key |
| 404 | Job không tồn tại, đã lấy, hoặc quá 15 phút |
| 413 | Ảnh lớn hơn 30 MB |
| 422 | Tham số sai kiểu hoặc ngoài giới hạn |
| 503 | Dịch vụ đang khởi động, hoặc chức năng chưa bật |
| 500 / 504 | Sinh ảnh lỗi / quá 10 phút |

## Viết prompt

- Viết tiếng Anh. Đặt phong cách ở đầu prompt (vd `digital painting, semi-realistic, ...`).
- Không nhắc tới bảng hiệu, biển, chữ, kể cả để cấm: càng nhắc model càng vẽ chữ.
- Tránh cụm từ dễ hiểu nghĩa đen (`red-rimmed eyes` thành dải đỏ che mắt).
- Không mô tả `left half` / `right half`: ảnh sẽ bị chia 2 khung.
- Với `/img2img` và `/control`, mô tả đủ mọi người và vật cần giữ.
- Chất lượng giữa các seed chênh nhau: dùng `n=3-4` rồi chọn.

## Lưu ý vận hành

- Một GPU xử lý lần lượt; request gửi cùng lúc xếp hàng.
- Đổi chức năng lần đầu chậm thêm vài giây để đổi model trong GPU.
- Thiếu VRAM/RAM: `TEXT_ENCODER_NAME=qwen_3_4b_fp8_mixed.safetensors` hoặc `COMFY_ARGS=--lowvram` trong `.env`.
- Log: `journalctl -u image-gen` và `logs/comfyui.log`.
- Giấy phép: Z-Image-Turbo và ControlNet là Apache 2.0. FLUX.1 Kontext dev chỉ dùng phi thương mại.
