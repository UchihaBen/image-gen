"""HTTP API for image generation, backed by a ComfyUI instance on localhost.

Endpoints
  POST /generate  text -> image                    (Z-Image-Turbo)
  POST /img2img   image + prompt -> variation      (Z-Image-Turbo)
  POST /edit      image + instruction -> edited    (FLUX.1 Kontext dev)
  POST /control   image (edges/pose/depth) + prompt -> new image with that structure
                                                   (Z-Image-Turbo + Fun ControlNet Union)
  GET  /jobs/{id} result of a request that runs as a job
  GET  /loras     style LoRAs installed on the server
  GET  /health

ComfyUI does the GPU work and only listens on 127.0.0.1. Images are returned to the
caller and not kept: uploaded inputs and ComfyUI's temp outputs are deleted after each
request, and unfetched job results live only in memory for JOB_TTL seconds.

Every request runs as a job. With wait=true (default) the request waits up to SYNC_WAIT
seconds for it; if it is not done by then the server answers 202 with the job id, so a
long queue never runs into the proxy's timeout.
"""
import asyncio
import base64
import contextvars
import io
import os
import random
import secrets
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = Path(os.environ.get("COMFY_DIR", ROOT / "ComfyUI"))
MODELS_DIR = COMFY_DIR / "models"
LORA_DIR = MODELS_DIR / "loras"
API_KEY = os.environ.get("API_KEY", "")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
GEN_TIMEOUT = float(os.environ.get("GEN_TIMEOUT", "600"))
SYNC_WAIT = float(os.environ.get("SYNC_WAIT", "85"))
JOB_TTL = float(os.environ.get("JOB_TTL", "900"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "30"))
UPLOAD_SUBFOLDER = "api"
CLIENT_ID = uuid.uuid4().hex
OUTPUT_NODE = "10"
MAX_SEED = 2**63 - 1

# Model files (names inside ComfyUI/models/<folder>/)
Z_UNET = os.environ.get("UNET_NAME", "z_image_turbo_bf16.safetensors")
Z_TEXT_ENCODER = os.environ.get("TEXT_ENCODER_NAME", "qwen_3_4b.safetensors")
VAE_NAME = os.environ.get("VAE_NAME", "ae.safetensors")
KONTEXT_UNET = os.environ.get("KONTEXT_UNET_NAME", "flux1-dev-kontext_fp8_scaled.safetensors")
CLIP_L = os.environ.get("CLIP_L_NAME", "clip_l.safetensors")
T5 = os.environ.get("T5_NAME", "t5xxl_fp8_e4m3fn_scaled.safetensors")
CONTROLNET = os.environ.get("CONTROLNET_NAME", "Z-Image-Turbo-Fun-Controlnet-Union.safetensors")

_BASE_FILES = [("diffusion_models", Z_UNET), ("text_encoders", Z_TEXT_ENCODER), ("vae", VAE_NAME)]
FEATURES = {
    "generate": (_BASE_FILES, ""),
    "img2img": (_BASE_FILES, ""),
    "edit": ([("diffusion_models", KONTEXT_UNET), ("text_encoders", CLIP_L),
              ("text_encoders", T5), ("vae", VAE_NAME)], "edit"),
    "control": (_BASE_FILES + [("model_patches", CONTROLNET)], "control"),
}
FORMATS = {"png": ("PNG", "image/png"), "jpeg": ("JPEG", "image/jpeg"), "webp": ("WEBP", "image/webp")}
Format = Literal["png", "jpeg", "webp"]

app = FastAPI(title="Image Generation API", version="3.0")
JOBS: dict[str, dict] = {}
CURRENT_JOB: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("current_job", default=None)


# ---------------------------------------------------------------- helpers

def require_api_key(request: Request):
    if not API_KEY:
        return
    key = request.headers.get("x-api-key", "")
    auth = request.headers.get("authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    if not secrets.compare_digest(key.encode(), API_KEY.encode()):
        raise HTTPException(401, "Invalid or missing API key")


def feature_ready(name: str) -> bool:
    files, _ = FEATURES[name]
    return all((MODELS_DIR / folder / filename).is_file() for folder, filename in files)


def require_feature(name: str):
    if not feature_ready(name):
        _, extra = FEATURES[name]
        print(f">> '{name}' requested but not installed. To enable: bash download_models.sh {extra}".rstrip(),
              flush=True)
        raise HTTPException(503, f"Feature '{name}' is not enabled on this server.")


def installed_loras() -> list[str]:
    if not LORA_DIR.is_dir():
        return []
    return sorted(p.name for p in LORA_DIR.iterdir() if p.suffix == ".safetensors")


def resolve_lora(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    available = installed_loras()
    for candidate in (name, f"{name}.safetensors"):
        if candidate in available:
            return candidate
    raise HTTPException(400, f"Unknown LoRA '{name}'. Available: {', '.join(available) or 'none'} (see GET /loras)")


def seeds_for(seed: int, n: int) -> list[int]:
    # Random seeds stay below 2**53 so JavaScript clients can read them back exactly.
    base = seed if seed >= 0 else random.randint(0, 2**53 - 1 - n)
    return [(base + i) % (MAX_SEED + 1) for i in range(n)]


def ceil16(x: int) -> int:
    return (x + 15) // 16 * 16


def cover_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to cover `size`, then center-crop to exactly `size`."""
    w, h = img.size
    tw, th = size
    if (w, h) == (tw, th):
        return img
    scale = max(tw / w, th / h)
    if abs(scale - 1) > 1e-6:
        img = img.resize((max(tw, round(w * scale)), max(th, round(h * scale))), Image.LANCZOS)
        w, h = img.size
    left, top = (w - tw) // 2, (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


def fit_image(img: Image.Image, megapixels: float) -> Image.Image:
    """Resize to about `megapixels` keeping the aspect ratio, sides multiples of 16."""
    w, h = img.size
    scale = (megapixels * 1024 * 1024 / (w * h)) ** 0.5
    nw = min(2048, max(256, round(w * scale / 16) * 16))
    nh = min(2048, max(256, round(h * scale / 16) * 16))
    return img.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (w, h) else img


def encode_output(png: bytes, size: Optional[tuple[int, int]], fmt: str, quality: int) -> dict:
    img = Image.open(io.BytesIO(png))
    changed = False
    if size and img.size != size:
        img, changed = cover_crop(img, size), True
    if fmt == "png" and not changed:
        data = png
    else:
        buf = io.BytesIO()
        pil_fmt, _ = FORMATS[fmt]
        if pil_fmt == "PNG":
            img.save(buf, "PNG")
        else:
            img.convert("RGB").save(buf, pil_fmt, quality=quality)
        data = buf.getvalue()
    return {"data": data, "width": img.size[0], "height": img.size[1]}


async def read_upload(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_MB:.0f} MB")
    try:
        img = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, f"Could not read '{file.filename}' as an image (use PNG, JPEG or WebP)")


def prepare_source(img: Image.Image, width: int, height: int, megapixels: float):
    """Returns (image to feed the model, exact output size or None)."""
    if width or height:
        if not (width and height):
            raise HTTPException(400, "Send both width and height, or neither")
        return cover_crop(img, (ceil16(width), ceil16(height))), (width, height)
    return fit_image(img, megapixels), None


def _remove_quietly(path: Path, root: Path):
    try:
        path = path.resolve()
        if root.resolve() in path.parents:
            path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------- ComfyUI calls

async def _upload(client: httpx.AsyncClient, img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    name = f"{uuid.uuid4().hex}.png"
    resp = await client.post("/upload/image",
                             files={"image": (name, buf.getvalue(), "image/png")},
                             data={"type": "input", "subfolder": UPLOAD_SUBFOLDER, "overwrite": "true"})
    if resp.status_code != 200:
        raise HTTPException(502, f"ComfyUI refused the uploaded image: {resp.text[:500]}")
    info = resp.json()
    return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]


async def _run_prompt(client: httpx.AsyncClient, workflow: dict) -> bytes:
    resp = await client.post("/prompt", json={"prompt": workflow, "client_id": CLIENT_ID})
    if resp.status_code != 200:
        raise HTTPException(502, f"ComfyUI rejected the workflow: {resp.text[:2000]}")
    prompt_id = resp.json()["prompt_id"]
    job = CURRENT_JOB.get()
    if job is not None:
        job["prompt_id"] = prompt_id

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
                if img.get("type") == "temp":
                    _remove_quietly(COMFY_DIR / "temp" / img.get("subfolder", "") / img["filename"],
                                    COMFY_DIR / "temp")
                return view.content
            if status.get("completed"):
                raise HTTPException(500, "ComfyUI finished but returned no image")
        if time.monotonic() > deadline:
            await client.post("/queue", json={"delete": [prompt_id]})
            raise HTTPException(504, f"Generation timed out after {GEN_TIMEOUT:.0f}s")
        await asyncio.sleep(0.3)


async def execute(build: Callable[..., dict], seeds: list[int], images: list[Image.Image] = (),
                  size: Optional[tuple[int, int]] = None, fmt: str = "png", quality: int = 90) -> list[dict]:
    """Upload the inputs once, run one workflow per seed, encode outputs, clean up."""
    names: list[str] = []
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(base_url=COMFY_URL, timeout=60) as client:
            try:
                for img in images:
                    names.append(await _upload(client, img))
                for seed in seeds:
                    png = await _run_prompt(client, build(seed, *names))
                    results.append({"seed": seed, **encode_output(png, size, fmt, quality)})
                    job = CURRENT_JOB.get()
                    if job is not None:
                        job["done_images"] = len(results)
            except httpx.HTTPError as e:
                raise HTTPException(503, f"Image backend is not reachable: {e}")
    finally:
        for name in names:
            _remove_quietly(COMFY_DIR / "input" / name, COMFY_DIR / "input")
    return results


async def comfy_queue() -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            return (await client.get(f"{COMFY_URL}/queue")).json()
    except Exception:
        return None


# ---------------------------------------------------------------- workflows (Z-Image-Turbo)

def _z_workflow(prompt: str, negative: str, cfg: float, shift: float, lora: Optional[str],
                lora_strength: float, controlnet: Optional[dict] = None) -> dict:
    """Loaders + model chain + conditioning. Model ends at node 4, positive 5, negative 6."""
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": Z_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": Z_TEXT_ENCODER, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
    }
    model = ["1", 0]
    if lora:
        wf["30"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model, "lora_name": lora, "strength_model": lora_strength}}
        model = ["30", 0]
    if controlnet:
        wf["20"] = {"class_type": "ModelPatchLoader", "inputs": {"name": CONTROLNET}}
        wf["22"] = {"class_type": "QwenImageDiffsynthControlnet", "inputs": {
            "model": model, "model_patch": ["20", 0], "vae": ["3", 0],
            "image": controlnet["image"], "strength": controlnet["strength"]}}
        model = ["22", 0]
    wf["4"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": shift}}
    if cfg > 1.0 and negative.strip():
        wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}}
    else:
        wf["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}}
    return wf


def _sample(wf: dict, node: str, latent, seed: int, steps: int, cfg: float, sampler: str,
            denoise: float = 1.0, model=("4", 0), positive=("5", 0)):
    wf[node] = {"class_type": "KSampler", "inputs": {
        "model": list(model), "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": "simple", "positive": list(positive),
        "negative": ["6", 0], "latent_image": list(latent), "denoise": denoise}}


def _decode_and_output(wf: dict, samples_node: str, decode_node: str = "9"):
    wf[decode_node] = {"class_type": "VAEDecode", "inputs": {"samples": [samples_node, 0], "vae": ["3", 0]}}
    wf[OUTPUT_NODE] = {"class_type": "PreviewImage", "inputs": {"images": [decode_node, 0]}}


def wf_generate(p: dict, seed: int) -> dict:
    wf = _z_workflow(p["prompt"], p["negative_prompt"], p["cfg"], p["shift"], p["lora"], p["lora_strength"])
    gw, gh = ceil16(p["width"]), ceil16(p["height"])
    if p["hires"]:
        bw = max(256, ceil16(round(gw / p["hires_scale"])))
        bh = max(256, ceil16(round(gh / p["hires_scale"])))
        wf["7"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": bw, "height": bh, "batch_size": 1}}
        _sample(wf, "8", ("7", 0), seed, p["steps"], p["cfg"], "res_multistep")
        wf["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}
        wf["40"] = {"class_type": "ImageScale", "inputs": {
            "image": ["9", 0], "upscale_method": "lanczos", "width": gw, "height": gh, "crop": "disabled"}}
        wf["41"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["40", 0], "vae": ["3", 0]}}
        _sample(wf, "42", ("41", 0), seed, p["steps"], p["cfg"], "res_multistep", denoise=p["hires_denoise"])
        _decode_and_output(wf, "42", "43")
    else:
        wf["7"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": gw, "height": gh, "batch_size": 1}}
        _sample(wf, "8", ("7", 0), seed, p["steps"], p["cfg"], "res_multistep")
        _decode_and_output(wf, "8")
    return wf


def wf_img2img(p: dict, seed: int, image: str) -> dict:
    wf = _z_workflow(p["prompt"], p["negative_prompt"], p["cfg"], p["shift"], p["lora"], p["lora_strength"])
    wf["11"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    wf["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["3", 0]}}
    _sample(wf, "8", ("12", 0), seed, p["steps"], p["cfg"], "res_multistep", denoise=p["strength"])
    _decode_and_output(wf, "8")
    return wf


def wf_control(p: dict, seed: int, image: str) -> dict:
    control_image = ["11", 0]
    extra = {"11": {"class_type": "LoadImage", "inputs": {"image": image}}}
    if p["mode"] == "canny":
        extra["21"] = {"class_type": "Canny", "inputs": {
            "image": ["11", 0], "low_threshold": p["canny_low"], "high_threshold": p["canny_high"]}}
        control_image = ["21", 0]
    wf = _z_workflow(p["prompt"], p["negative_prompt"], p["cfg"], p["shift"], p["lora"], p["lora_strength"],
                     controlnet={"image": control_image, "strength": p["strength"]})
    wf.update(extra)
    wf["7"] = {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": p["width"], "height": p["height"], "batch_size": 1}}
    _sample(wf, "8", ("7", 0), seed, p["steps"], p["cfg"], "res_multistep")
    _decode_and_output(wf, "8")
    return wf


def wf_edit(p: dict, seed: int, image: str, image2: Optional[str] = None) -> dict:
    # Mirrors ComfyUI's official "flux_kontext_dev_basic" template.
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KONTEXT_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": CLIP_L, "clip_name2": T5, "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["2", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image}},
        "13": {"class_type": "ImageStitch", "inputs": {
            "image1": ["11", 0], "direction": "right", "match_image_size": True,
            "spacing_width": 0, "spacing_color": "white"}},
        "14": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["13", 0]}},
        "15": {"class_type": "VAEEncode", "inputs": {"pixels": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["5", 0], "latent": ["15", 0]}},
        "17": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["16", 0], "guidance": p["guidance"]}},
    }
    if image2:
        wf["12"] = {"class_type": "LoadImage", "inputs": {"image": image2}}
        wf["13"]["inputs"]["image2"] = ["12", 0]
    _sample(wf, "8", ("15", 0), seed, p["steps"], 1.0, "euler", model=("1", 0), positive=("17", 0))
    _decode_and_output(wf, "8")
    return wf


# ---------------------------------------------------------------- responses and jobs

def result_response(results: list[dict], elapsed: float, fmt: str, as_base64: bool):
    if len(results) == 1 and not as_base64:
        r = results[0]
        return Response(r["data"], media_type=FORMATS[fmt][1], headers={
            "X-Seed": str(r["seed"]), "X-Elapsed-S": f"{elapsed:.2f}",
            "X-Image-Width": str(r["width"]), "X-Image-Height": str(r["height"])})
    images = [{"seed": r["seed"], "width": r["width"], "height": r["height"], "format": fmt,
               "image_base64": base64.b64encode(r["data"]).decode()} for r in results]
    body = {"elapsed_s": round(elapsed, 2), "format": fmt, "count": len(images), "images": images}
    if len(images) == 1:  # keep the v2 shape for single images
        body.update(seed=images[0]["seed"], image_base64=images[0]["image_base64"])
    return JSONResponse(body)


def _purge_jobs():
    now = time.time()
    for job_id in [j for j, v in JOBS.items() if v["status"] != "running" and now - v["created"] > JOB_TTL]:
        JOBS.pop(job_id, None)


async def _run_job(job: dict, work: Callable[[], Awaitable[list[dict]]]):
    CURRENT_JOB.set(job)
    started = time.monotonic()
    try:
        job["results"] = await work()
        job["status"] = "done"
    except HTTPException as e:
        job.update(status="error", error=e.detail, code=e.status_code)
    except Exception as e:  # keep the job record instead of losing the error
        job.update(status="error", error=str(e), code=500)
    job["elapsed"] = time.monotonic() - started


async def _job_status(job: dict) -> dict:
    info = {"job_id": job["id"], "status": job["status"], "kind": job["kind"],
            "seeds": job["seeds"], "images_done": job.get("done_images", 0),
            "elapsed_s": round(time.time() - job["created"], 1),
            "result_url": f"/jobs/{job['id']}"}
    queue = await comfy_queue()
    pid = job.get("prompt_id")
    if queue is not None:
        running = [item[1] for item in queue.get("queue_running", [])]
        pending = [item[1] for item in sorted(queue.get("queue_pending", []), key=lambda i: i[0])]
        if pid in running:
            info["queue_position"] = 0
        elif pid in pending:
            info["queue_position"] = pending.index(pid) + 1
        else:  # not submitted to the GPU yet: everything queued is ahead of it
            info["queue_position"] = len(running) + len(pending) + 1 if not pid else 0
    return info


async def dispatch(kind: str, seeds: list[int], fmt: str, work: Callable[[], Awaitable[list[dict]]],
                   wait: bool, as_base64: bool):
    _purge_jobs()
    job = {"id": uuid.uuid4().hex, "status": "running", "kind": kind, "seeds": seeds,
           "format": fmt, "created": time.time()}
    JOBS[job["id"]] = job
    job["task"] = asyncio.create_task(_run_job(job, work))
    if wait:
        try:
            await asyncio.wait_for(asyncio.shield(job["task"]), SYNC_WAIT)
        except asyncio.TimeoutError:
            info = await _job_status(job)
            info["note"] = f"Still running after {SYNC_WAIT:.0f}s; fetch the result from result_url"
            return JSONResponse(info, status_code=202)
        JOBS.pop(job["id"], None)
        if job["status"] == "error":
            raise HTTPException(job.get("code", 500), job["error"])
        return result_response(job["results"], job["elapsed"], fmt, as_base64)
    return JSONResponse(await _job_status(job), status_code=202)


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str, as_base64: bool = False):
    _purge_jobs()
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown or expired job (results are kept for "
                                 f"{JOB_TTL / 60:.0f} minutes and deleted once fetched)")
    if job["status"] == "running":
        return JSONResponse(await _job_status(job), status_code=202)
    JOBS.pop(job_id, None)
    if job["status"] == "error":
        return JSONResponse({"job_id": job_id, "status": "error", "detail": job["error"]},
                            status_code=job.get("code", 500))
    return result_response(job["results"], job["elapsed"], job["format"], as_base64)


# ---------------------------------------------------------------- endpoints

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="Mô tả ảnh (tiếng Anh cho kết quả tốt nhất)")
    width: int = Field(1024, ge=256, le=2048, description="Kích thước ảnh ra, đúng từng pixel (vd 1920)")
    height: int = Field(1024, ge=256, le=2048, description="Kích thước ảnh ra, đúng từng pixel (vd 1080)")
    n: int = Field(1, ge=1, le=4, description="Số ảnh, mỗi ảnh một seed liên tiếp: seed, seed+1, ...")
    steps: int = Field(8, ge=1, le=50)
    cfg: float = Field(1.0, ge=1.0, le=10.0, description="1.0 = bỏ qua negative_prompt (nhanh nhất)")
    negative_prompt: str = Field("", max_length=2000, description="Chỉ có tác dụng khi cfg > 1 (thử nghiệm)")
    shift: float = Field(3.0, ge=0.0, le=20.0)
    seed: int = Field(-1, ge=-1, le=MAX_SEED, description="-1 = ngẫu nhiên")
    lora: Optional[str] = Field(None, description="Tên LoRA phong cách, xem GET /loras")
    lora_strength: float = Field(0.8, ge=0.0, le=2.0)
    hires: bool = Field(False, description="Gen 2 lượt: nháp nhỏ rồi phóng to và tinh chỉnh")
    hires_scale: float = Field(1.5, ge=1.1, le=2.0)
    hires_denoise: float = Field(0.35, ge=0.1, le=0.8)
    format: Format = "png"
    quality: int = Field(90, ge=50, le=100, description="Cho jpeg/webp")
    as_base64: bool = Field(False, description="true: trả JSON có ảnh base64")
    wait: bool = Field(True, description="false: trả job_id ngay, lấy ảnh ở GET /jobs/{job_id}")


@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(req: GenerateRequest):
    require_feature("generate")
    p = req.model_dump()
    p["lora"] = resolve_lora(req.lora)
    seeds = seeds_for(req.seed, req.n)
    size = (req.width, req.height)
    work = lambda: execute(lambda s: wf_generate(p, s), seeds, size=size, fmt=req.format, quality=req.quality)
    return await dispatch("generate", seeds, req.format, work, req.wait, req.as_base64)


@app.post("/img2img", dependencies=[Depends(require_api_key)])
async def img2img(
    image: UploadFile = File(..., description="Ảnh mẫu"),
    prompt: str = Form(..., min_length=1, max_length=4000),
    strength: float = Form(0.6, ge=0.05, le=1.0, description="Mức thay đổi: 0.3 giữ gần như nguyên, 0.8 vẽ lại nhiều"),
    width: int = Form(0, ge=0, le=2048, description="Kích thước ảnh ra (0 = theo megapixels)"),
    height: int = Form(0, ge=0, le=2048),
    megapixels: float = Form(1.0, ge=0.25, le=2.0, description="Dùng khi không gửi width/height"),
    n: int = Form(1, ge=1, le=4),
    steps: int = Form(8, ge=1, le=50),
    cfg: float = Form(1.0, ge=1.0, le=10.0),
    negative_prompt: str = Form("", max_length=2000),
    shift: float = Form(3.0, ge=0.0, le=20.0),
    seed: int = Form(-1, ge=-1, le=MAX_SEED),
    lora: Optional[str] = Form(None),
    lora_strength: float = Form(0.8, ge=0.0, le=2.0),
    format: Format = Form("png"),
    quality: int = Form(90, ge=50, le=100),
    as_base64: bool = Form(False),
    wait: bool = Form(True),
):
    require_feature("img2img")
    if width and width < 256 or height and height < 256:
        raise HTTPException(400, "width and height must be at least 256")
    src, size = prepare_source(await read_upload(image), width, height, megapixels)
    p = dict(prompt=prompt, strength=strength, steps=steps, cfg=cfg, negative_prompt=negative_prompt,
             shift=shift, lora=resolve_lora(lora), lora_strength=lora_strength)
    seeds = seeds_for(seed, n)
    work = lambda: execute(lambda s, name: wf_img2img(p, s, name), seeds, [src], size, format, quality)
    return await dispatch("img2img", seeds, format, work, wait, as_base64)


@app.post("/control", dependencies=[Depends(require_api_key)])
async def control(
    image: UploadFile = File(..., description="Ảnh mẫu (canny) hoặc ảnh điều khiển có sẵn: pose, depth, lineart (direct)"),
    prompt: str = Form(..., min_length=1, max_length=4000),
    mode: str = Form("canny", pattern="^(canny|direct)$",
                     description="canny: tự tách đường nét từ ảnh; direct: dùng ảnh gửi lên làm ảnh điều khiển"),
    strength: float = Form(0.9, ge=0.0, le=2.0),
    canny_low: float = Form(0.1, ge=0.01, le=0.99),
    canny_high: float = Form(0.32, ge=0.01, le=0.99),
    width: int = Form(0, ge=0, le=2048, description="Kích thước ảnh ra (0 = theo megapixels)"),
    height: int = Form(0, ge=0, le=2048),
    megapixels: float = Form(1.0, ge=0.25, le=2.0),
    n: int = Form(1, ge=1, le=4),
    steps: int = Form(8, ge=1, le=50),
    cfg: float = Form(1.0, ge=1.0, le=10.0),
    negative_prompt: str = Form("", max_length=2000),
    shift: float = Form(3.0, ge=0.0, le=20.0),
    seed: int = Form(-1, ge=-1, le=MAX_SEED),
    lora: Optional[str] = Form(None),
    lora_strength: float = Form(0.8, ge=0.0, le=2.0),
    format: Format = Form("png"),
    quality: int = Form(90, ge=50, le=100),
    as_base64: bool = Form(False),
    wait: bool = Form(True),
):
    require_feature("control")
    if canny_low >= canny_high:
        raise HTTPException(400, "canny_low must be smaller than canny_high")
    if width and width < 256 or height and height < 256:
        raise HTTPException(400, "width and height must be at least 256")
    src, size = prepare_source(await read_upload(image), width, height, megapixels)
    p = dict(prompt=prompt, mode=mode, strength=strength, canny_low=canny_low, canny_high=canny_high,
             width=src.size[0], height=src.size[1], steps=steps, cfg=cfg, negative_prompt=negative_prompt,
             shift=shift, lora=resolve_lora(lora), lora_strength=lora_strength)
    seeds = seeds_for(seed, n)
    work = lambda: execute(lambda s, name: wf_control(p, s, name), seeds, [src], size, format, quality)
    return await dispatch("control", seeds, format, work, wait, as_base64)


@app.post("/edit", dependencies=[Depends(require_api_key)])
async def edit(
    image: UploadFile = File(..., description="Ảnh cần sửa"),
    prompt: str = Form(..., min_length=1, max_length=4000, description="Câu lệnh sửa, tiếng Anh"),
    image2: Optional[UploadFile] = File(None, description="Ảnh tham chiếu thứ hai (tuỳ chọn), ghép bên phải ảnh 1"),
    guidance: float = Form(2.5, ge=1.0, le=10.0),
    megapixels: float = Form(1.0, ge=0.25, le=2.0),
    n: int = Form(1, ge=1, le=4),
    steps: int = Form(20, ge=1, le=50),
    seed: int = Form(-1, ge=-1, le=MAX_SEED),
    format: Format = Form("png"),
    quality: int = Form(90, ge=50, le=100),
    as_base64: bool = Form(False),
    wait: bool = Form(True),
):
    require_feature("edit")
    images = [fit_image(await read_upload(image), megapixels)]
    if image2 is not None and image2.filename:
        images.append(fit_image(await read_upload(image2), megapixels))
    p = dict(prompt=prompt, guidance=guidance, steps=steps)
    seeds = seeds_for(seed, n)
    work = lambda: execute(lambda s, *names: wf_edit(p, s, *names), seeds, images, None, format, quality)
    return await dispatch("edit", seeds, format, work, wait, as_base64)


@app.get("/loras", dependencies=[Depends(require_api_key)])
async def loras():
    return {"loras": installed_loras()}


@app.get("/", response_class=HTMLResponse)
async def test_page():
    return (ROOT / "test_page.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    features = {name: feature_ready(name) for name in FEATURES}
    jobs_running = sum(1 for j in JOBS.values() if j["status"] == "running")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            stats = (await client.get(f"{COMFY_URL}/system_stats")).json()
            queue = (await client.get(f"{COMFY_URL}/queue")).json()
    except Exception:
        return JSONResponse({"ok": False, "backend": "down", "features": features,
                             "jobs_pending": jobs_running}, status_code=503)
    devices = [{"name": d.get("name"), "vram_total_gb": round(d.get("vram_total", 0) / 2**30, 1),
                "vram_free_gb": round(d.get("vram_free", 0) / 2**30, 1)} for d in stats.get("devices", [])]
    return {"ok": True, "backend": "up", "auth_required": bool(API_KEY), "features": features,
            "jobs_pending": jobs_running,
            "gpu_queue": {"running": len(queue.get("queue_running", [])),
                          "waiting": len(queue.get("queue_pending", []))},
            "loras": len(installed_loras()), "devices": devices}


@app.on_event("startup")
async def _schedule_warmup():
    # Load the base model before the first real request. Behind Cloudflare a cold
    # first request can exceed the 100 s proxy timeout.
    if os.environ.get("WARMUP", "1") == "1" and feature_ready("generate"):
        asyncio.create_task(_warmup())


async def _warmup():
    started = time.monotonic()
    p = dict(prompt="a simple test image", negative_prompt="", cfg=1.0, shift=3.0, lora=None, lora_strength=0.0,
             width=256, height=256, steps=1, hires=False)
    try:
        await execute(lambda s: wf_generate(p, s), [0])
        print(f">> Warm-up done in {time.monotonic() - started:.0f}s, models are loaded", flush=True)
    except Exception as e:
        print(f">> Warm-up failed: {getattr(e, 'detail', e)}", flush=True)


if __name__ == "__main__":
    import uvicorn

    if not API_KEY and HOST not in ("127.0.0.1", "localhost"):
        raise SystemExit("API_KEY is empty but HOST is public. Set API_KEY in .env first.")
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=30)
