"""HTTP API for Z-Image-Turbo text-to-image, backed by a ComfyUI instance on localhost.

ComfyUI does the GPU work and only listens on 127.0.0.1. This server is the only thing
exposed to the network: it checks the API key, builds the workflow, waits for ComfyUI
and returns the image.
"""
import asyncio
import base64
import os
import random
import secrets
import time
import uuid
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_TEMP_DIR = Path(os.environ.get("COMFY_TEMP_DIR", ROOT / "ComfyUI" / "temp"))
API_KEY = os.environ.get("API_KEY", "")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
UNET_NAME = os.environ.get("UNET_NAME", "z_image_turbo_bf16.safetensors")
TEXT_ENCODER_NAME = os.environ.get("TEXT_ENCODER_NAME", "qwen_3_4b.safetensors")
VAE_NAME = os.environ.get("VAE_NAME", "ae.safetensors")
GEN_TIMEOUT = float(os.environ.get("GEN_TIMEOUT", "600"))
CLIENT_ID = uuid.uuid4().hex
OUTPUT_NODE = "10"

app = FastAPI(title="Z-Image-Turbo API", version="1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="Mô tả ảnh (tiếng Anh cho kết quả tốt nhất)")
    width: int = Field(1024, ge=256, le=2048, description="Bội số của 16")
    height: int = Field(1024, ge=256, le=2048, description="Bội số của 16")
    steps: int = Field(8, ge=1, le=50)
    cfg: float = Field(1.0, ge=0.0, le=10.0)
    shift: float = Field(3.0, ge=0.0, le=20.0)
    seed: int = Field(-1, ge=-1, le=2**63 - 1, description="-1 = ngẫu nhiên")
    as_base64: bool = Field(False, description="true: trả JSON có ảnh base64 thay vì file PNG")


def require_api_key(request: Request):
    if not API_KEY:
        return
    key = request.headers.get("x-api-key", "")
    auth = request.headers.get("authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    if not secrets.compare_digest(key.encode(), API_KEY.encode()):
        raise HTTPException(401, "Invalid or missing API key")


def build_workflow(req: GenerateRequest, seed: int) -> dict:
    # Mirrors ComfyUI's official "image_z_image_turbo" template.
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER_NAME, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": req.shift}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": req.prompt, "clip": ["2", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": req.width, "height": req.height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["4", 0], "seed": seed, "steps": req.steps, "cfg": req.cfg,
            "sampler_name": "res_multistep", "scheduler": "simple",
            "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        # PreviewImage writes to ComfyUI/temp; we delete the file after reading it.
        OUTPUT_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
    }


async def run_workflow(workflow: dict) -> bytes:
    async with httpx.AsyncClient(base_url=COMFY_URL, timeout=30) as client:
        try:
            resp = await client.post("/prompt", json={"prompt": workflow, "client_id": CLIENT_ID})
        except httpx.HTTPError as e:
            raise HTTPException(503, f"ComfyUI is not reachable at {COMFY_URL}: {e}")
        if resp.status_code != 200:
            raise HTTPException(502, f"ComfyUI rejected the workflow: {resp.text[:2000]}")
        prompt_id = resp.json()["prompt_id"]

        deadline = time.monotonic() + GEN_TIMEOUT
        while True:
            hist = (await client.get(f"/history/{prompt_id}")).json().get(prompt_id)
            if hist:
                status = hist.get("status", {})
                if status.get("status_str") == "error":
                    raise HTTPException(500, f"Generation failed: {status.get('messages')}")
                images = hist.get("outputs", {}).get(OUTPUT_NODE, {}).get("images", [])
                if images:
                    img = images[0]
                    view = await client.get("/view", params={
                        "filename": img["filename"], "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "temp")})
                    view.raise_for_status()
                    _cleanup_temp(img)
                    return view.content
                if status.get("completed"):
                    raise HTTPException(500, "ComfyUI finished but returned no image")
            if time.monotonic() > deadline:
                await client.post("/queue", json={"delete": [prompt_id]})
                raise HTTPException(504, f"Generation timed out after {GEN_TIMEOUT:.0f}s")
            await asyncio.sleep(0.3)


def _cleanup_temp(img: dict):
    if img.get("type") != "temp":
        return
    try:
        path = (COMFY_TEMP_DIR / img.get("subfolder", "") / img["filename"]).resolve()
        if COMFY_TEMP_DIR.resolve() in path.parents:
            path.unlink(missing_ok=True)
    except OSError:
        pass


@app.on_event("startup")
async def _schedule_warmup():
    # Load the models into memory before the first real request. Behind Cloudflare
    # a cold first request can exceed the 100 s proxy timeout.
    if os.environ.get("WARMUP", "1") == "1":
        asyncio.create_task(_warmup())


async def _warmup():
    started = time.monotonic()
    try:
        req = GenerateRequest(prompt="a simple test image", width=256, height=256, steps=1)
        await run_workflow(build_workflow(req, seed=0))
        print(f">> Warm-up done in {time.monotonic() - started:.0f}s, models are loaded", flush=True)
    except Exception as e:
        print(f">> Warm-up failed: {getattr(e, 'detail', e)}", flush=True)


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            stats = (await client.get(f"{COMFY_URL}/system_stats")).json()
    except Exception as e:
        return JSONResponse({"ok": False, "comfyui": f"unreachable: {e}"}, status_code=503)
    devices = [{"name": d.get("name"), "vram_total_gb": round(d.get("vram_total", 0) / 2**30, 1),
                "vram_free_gb": round(d.get("vram_free", 0) / 2**30, 1)} for d in stats.get("devices", [])]
    return {"ok": True, "auth_required": bool(API_KEY), "model": UNET_NAME,
            "text_encoder": TEXT_ENCODER_NAME, "devices": devices}


@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(req: GenerateRequest):
    if req.width % 16 or req.height % 16:
        raise HTTPException(400, "width and height must be multiples of 16")
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**63 - 1)
    started = time.monotonic()
    png = await run_workflow(build_workflow(req, seed))
    elapsed = f"{time.monotonic() - started:.2f}"
    if req.as_base64:
        return {"seed": seed, "elapsed_s": float(elapsed), "image_base64": base64.b64encode(png).decode()}
    return Response(png, media_type="image/png", headers={"X-Seed": str(seed), "X-Elapsed-S": elapsed})


if __name__ == "__main__":
    import uvicorn

    if not API_KEY and HOST not in ("127.0.0.1", "localhost"):
        raise SystemExit("API_KEY is empty but HOST is public. Set API_KEY in .env first.")
    uvicorn.run(app, host=HOST, port=PORT)
