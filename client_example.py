"""Gọi API từ máy bất kỳ: mỗi dòng trong file prompt -> 1 ảnh (đánh số để ghép video).

    python client_example.py --url http://SERVER:8000 --key KEY --prompts prompts.txt --out frames
    python client_example.py --url http://SERVER:8000 --key KEY --prompt "a red fox in snow"

Chỉ dùng thư viện chuẩn của Python, không cần cài gì thêm.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def generate(url, key, prompt, width, height, seed):
    body = json.dumps({"prompt": prompt, "width": width, "height": height, "seed": seed}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/generate", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return resp.read(), resp.headers.get("X-Seed"), resp.headers.get("X-Elapsed-S")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--key", default="")
    p.add_argument("--prompt")
    p.add_argument("--prompts", help="file .txt, mỗi dòng một prompt")
    p.add_argument("--out", default="outputs")
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--seed", type=int, default=-1)
    a = p.parse_args()

    if a.prompts:
        prompts = [l.strip() for l in Path(a.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif a.prompt:
        prompts = [a.prompt]
    else:
        sys.exit("Cần --prompt hoặc --prompts")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts, 1):
        try:
            png, seed, elapsed = generate(a.url, a.key, prompt, a.width, a.height, a.seed)
        except urllib.error.HTTPError as e:
            print(f"[{i}/{len(prompts)}] LỖI {e.code}: {e.read().decode(errors='replace')[:300]}")
            continue
        path = out / f"{i:04d}.png"
        path.write_bytes(png)
        print(f"[{i}/{len(prompts)}] {path}  seed={seed}  {elapsed}s")


if __name__ == "__main__":
    main()
