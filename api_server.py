"""HTTP API for image generation, backed by a ComfyUI instance on localhost.

Endpoints
  POST /generate  text -> image                    (Z-Image-Turbo)
  POST /img2img   image + prompt -> variation      (Z-Image-Turbo)
  POST /edit      image + instruction -> edited    (FLUX.1 Kontext dev)
  POST /control   image (edges/pose/depth) + prompt -> new image with that structure
                                                   (Z-Image-Turbo + Fun ControlNet Union)
  GET  /jobs/{id} result of a request sent with wait=false
  GET  /health

ComfyUI does the GPU work and only listens on 127.0.0.1. Images are returned to the
caller and not kept: uploaded inputs and ComfyUI's temp outputs are deleted after each
request, and unfinished/unfetched job results live only in memory for JOB_TTL seconds.
"""
import asyncio
import base64
import io
import os
import random
import secrets
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = Path(os.environ.get("COMFY_DIR", ROOT / "ComfyUI"))
MODELS_DIR = COMFY_DIR / "models"
API_KEY = os.environ.get("API_KEY", "")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
GEN_TIMEOUT = float(os.environ.get("GEN_TIMEOUT", "600"))
JOB_TTL = float(os.environ.get("JOB_TTL", "900"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "20"))
UPLOAD_SUBFOLDER = "api"
CLIENT_ID = uuid.uuid4().hex
OUTPUT_NODE = "10"

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

app = FastAPI(title="Image Generation API", version="2.0")
JOBS: dict[str, dict] = {}


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
        raise HTTPException(503, f"Feature '{name}' is not installed. On the server run: "
                                 f"bash download_models.sh {extra}".strip())


def pick_seed(seed: int) -> int:
    return seed if seed >= 0 else random.randint(0, 2**63 - 1)


def fit_image(img: Image.Image, megapixels: float) -> Image.Image:
    """Resize to about `megapixels` keeping the aspect ratio, sides multiples of 16."""
    w, h = img.size
    scale = (megapixels * 1024 * 1024 / (w * h)) ** 0.5
    nw = min(2048, max(256, round(w * scale / 16) * 16))
    nh = min(2048, max(256, round(h * scale / 16) * 16))
    return img.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (w, h) else img


async def read_upload(file: UploadFile, megapixels: float) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_MB:.0f} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, f"Could not read '{file.filename}' as an image (use PNG, JPEG or WebP)")
    return fit_image(img, megapixels)


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


async def execute(build: Callable[..., dict], images: list[Image.Image] = ()) -> bytes:
    """Upload the input images, run the workflow built from their names, clean up."""
    names: list[str] = []
    try:
        async with httpx.AsyncClient(base_url=COMFY_URL, timeout=60) as client:
            try:
                for img in images:
                    names.append(await _upload(client, img))
                return await _run_prompt(client, build(*names))
            except httpx.HTTPError as e:
                raise HTTPException(503, f"ComfyUI is not reachable at {COMFY_URL}: {e}")
    finally:
        for name in names:
            _remove_quietly(COMFY_DIR / "input" / name, COMFY_DIR / "input")


# ---------------------------------------------------------------- workflows

def _z_base(prompt: str, shift: float, model_ref=("1", 0)) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": Z_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": Z_TEXT_ENCODER, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": list(model_ref), "shift": shift}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
    }


def _finish(wf: dict, *, latent, seed: int, steps: int, cfg: float, sampler: str,
            model=("4", 0), positive=("5", 0), denoise: float = 1.0) -> dict:
    wf["8"] = {"class_type": "KSampler", "inputs": {
        "model": list(model), "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler, "scheduler": "simple", "positive": list(positive),
        "negative": ["6", 0], "latent_image": list(latent), "denoise": denoise}}
    wf["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}
    wf[OUTPUT_NODE] = {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}}
    return wf


def wf_generate(prompt, width, height, seed, steps, cfg, shift):
    wf = _z_base(prompt, shift)
    wf["7"] = {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}}
    return _finish(wf, latent=("7", 0), seed=seed, steps=steps, cfg=cfg, sampler="res_multistep")


def wf_img2img(image, prompt, seed, steps, strength, shift):
    wf = _z_base(prompt, shift)
    wf["11"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    wf["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["3", 0]}}
    return _finish(wf, latent=("12", 0), seed=seed, steps=steps, cfg=1.0,
                   sampler="res_multistep", denoise=strength)


def wf_control(image, prompt, width, height, seed, steps, strength, shift, mode, low, high):
    wf = _z_base(prompt, shift, model_ref=("22", 0))
    wf["11"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    control = ["11", 0]
    if mode == "canny":
        wf["21"] = {"class_type": "Canny",
                    "inputs": {"image": ["11", 0], "low_threshold": low, "high_threshold": high}}
        control = ["21", 0]
    wf["20"] = {"class_type": "ModelPatchLoader", "inputs": {"name": CONTROLNET}}
    wf["22"] = {"class_type": "QwenImageDiffsynthControlnet", "inputs": {
        "model": ["1", 0], "model_patch": ["20", 0], "vae": ["3", 0], "image": control, "strength": strength}}
    wf["7"] = {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}}
    return _finish(wf, latent=("7", 0), seed=seed, steps=steps, cfg=1.0, sampler="res_multistep")


def wf_edit(image, image2, prompt, seed, steps, guidance):
    # Mirrors ComfyUI's official "flux_kontext_dev_basic" template.
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KONTEXT_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": CLIP_L, "clip_name2": T5, "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image}},
        "13": {"class_type": "ImageStitch", "inputs": {
            "image1": ["11", 0], "direction": "right", "match_image_size": True,
            "spacing_width": 0, "spacing_color": "white"}},
        "14": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["13", 0]}},
        "15": {"class_type": "VAEEncode", "inputs": {"pixels": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["5", 0], "latent": ["15", 0]}},
        "17": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["16", 0], "guidance": guidance}},
    }
    if image2:
        wf["12"] = {"class_type": "LoadImage", "inputs": {"image": image2}}
        wf["13"]["inputs"]["image2"] = ["12", 0]
    return _finish(wf, latent=("15", 0), seed=seed, steps=steps, cfg=1.0, sampler="euler",
                   model=("1", 0), positive=("17", 0))


# ---------------------------------------------------------------- responses and jobs

def image_response(png: bytes, seed: int, elapsed: float, as_base64: bool):
    if as_base64:
        return {"seed": seed, "elapsed_s": round(elapsed, 2), "image_base64": base64.b64encode(png).decode()}
    return Response(png, media_type="image/png",
                    headers={"X-Seed": str(seed), "X-Elapsed-S": f"{elapsed:.2f}"})


def _purge_jobs():
    now = time.time()
    for job_id in [j for j, v in JOBS.items() if now - v["created"] > JOB_TTL]:
        JOBS.pop(job_id, None)


async def _run_job(job_id: str, work: Callable[[], Awaitable[bytes]]):
    job = JOBS[job_id]
    started = time.monotonic()
    try:
        job["image"] = await work()
        job["status"] = "done"
    except HTTPException as e:
        job.update(status="error", error=e.detail, code=e.status_code)
    except Exception as e:  # keep the job record instead of losing the error
        job.update(status="error", error=str(e), code=500)
    job["elapsed"] = time.monotonic() - started
    job.pop("task", None)


async def dispatch(kind: str, seed: int, work: Callable[[], Awaitable[bytes]], wait: bool, as_base64: bool):
    if wait:
        started = time.monotonic()
        png = await work()
        return image_response(png, seed, time.monotonic() - started, as_base64)
    _purge_jobs()
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "kind": kind, "seed": seed, "created": time.time()}
    JOBS[job_id]["task"] = asyncio.create_task(_run_job(job_id, work))
    return JSONResponse({"job_id": job_id, "status": "running", "seed": seed,
                         "result_url": f"/jobs/{job_id}"}, status_code=202)


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str, as_base64: bool = False):
    _purge_jobs()
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown or expired job (results are kept for "
                                 f"{JOB_TTL / 60:.0f} minutes and deleted once fetched)")
    if job["status"] == "running":
        return JSONResponse({"job_id": job_id, "status": "running", "kind": job["kind"],
                             "elapsed_s": round(time.time() - job["created"], 1)}, status_code=202)
    JOBS.pop(job_id, None)
    if job["status"] == "error":
        return JSONResponse({"job_id": job_id, "status": "error", "detail": job["error"]},
                            status_code=job.get("code", 500))
    return image_response(job["image"], job["seed"], job["elapsed"], as_base64)


# ---------------------------------------------------------------- endpoints

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="Mô tả ảnh (tiếng Anh cho kết quả tốt nhất)")
    width: int = Field(1024, ge=256, le=2048, description="Bội số của 16")
    height: int = Field(1024, ge=256, le=2048, description="Bội số của 16")
    steps: int = Field(8, ge=1, le=50)
    cfg: float = Field(1.0, ge=0.0, le=10.0)
    shift: float = Field(3.0, ge=0.0, le=20.0)
    seed: int = Field(-1, ge=-1, le=2**63 - 1, description="-1 = ngẫu nhiên")
    as_base64: bool = Field(False, description="true: trả JSON có ảnh base64 thay vì file PNG")
    wait: bool = Field(True, description="false: trả job_id ngay, lấy ảnh ở GET /jobs/{job_id}")


@app.post("/generate", dependencies=[Depends(require_api_key)])
async def generate(req: GenerateRequest):
    require_feature("generate")
    if req.width % 16 or req.height % 16:
        raise HTTPException(400, "width and height must be multiples of 16")
    seed = pick_seed(req.seed)
    work = lambda: execute(lambda: wf_generate(req.prompt, req.width, req.height, seed, req.steps, req.cfg, req.shift))
    return await dispatch("generate", seed, work, req.wait, req.as_base64)


@app.post("/img2img", dependencies=[Depends(require_api_key)])
async def img2img(
    image: UploadFile = File(..., description="Ảnh mẫu"),
    prompt: str = Form(..., min_length=1, max_length=4000),
    strength: float = Form(0.6, ge=0.05, le=1.0, description="Mức thay đổi: 0.3 giữ gần như nguyên, 0.8 vẽ lại nhiều"),
    steps: int = Form(8, ge=1, le=50),
    shift: float = Form(3.0, ge=0.0, le=20.0),
    seed: int = Form(-1, ge=-1, le=2**63 - 1),
    megapixels: float = Form(1.0, ge=0.25, le=2.0, description="Kích thước ảnh ra, giữ tỉ lệ ảnh mẫu"),
    as_base64: bool = Form(False),
    wait: bool = Form(True),
):
    require_feature("img2img")
    src = await read_upload(image, megapixels)
    seed = pick_seed(seed)
    work = lambda: execute(lambda name: wf_img2img(name, prompt, seed, steps, strength, shift), [src])
    return await dispatch("img2img", seed, work, wait, as_base64)


@app.post("/edit", dependencies=[Depends(require_api_key)])
async def edit(
    image: UploadFile = File(..., description="Ảnh cần sửa"),
    prompt: str = Form(..., min_length=1, max_length=4000, description="Câu lệnh sửa, tiếng Anh"),
    image2: Optional[UploadFile] = File(None, description="Ảnh tham chiếu thứ hai (tuỳ chọn), ghép bên phải ảnh 1"),
    guidance: float = Form(2.5, ge=1.0, le=10.0),
    steps: int = Form(20, ge=1, le=50),
    seed: int = Form(-1, ge=-1, le=2**63 - 1),
    megapixels: float = Form(1.0, ge=0.25, le=2.0),
    as_base64: bool = Form(False),
    wait: bool = Form(True),
):
    require_feature("edit")
    images = [await read_upload(image, megapixels)]
    if image2 is not None and image2.filename:
        images.append(await read_upload(image2, megapixels))
    seed = pick_seed(seed)

    def build(name, name2=None):
        return wf_edit(name, name2, prompt, seed, steps, guidance)

    work = lambda: execute(build, images)
    return await dispatch("edit", seed, work, wait, as_base64)


@app.post("/control", dependencies=[Depends(require_api_key)])
async def control(
    image: UploadFile = File(..., description="Ảnh mẫu (canny) hoặc ảnh điều khiển có sẵn: pose, depth, lineart (direct)"),
    prompt: str = Form(..., min_length=1, max_length=4000),
    mode: str = Form("canny", pattern="^(canny|direct)$",
                     description="canny: tự tách đường nét từ ảnh; direct: dùng ảnh gửi lên làm ảnh điều khiển"),
    strength: float = Form(0.9, ge=0.0, le=2.0),
    canny_low: float = Form(0.1, ge=0.01, le=0.99),
    canny_high: float = Form(0.32, ge=0.01, le=0.99),
    steps: int = Form(8, ge=1, le=50),
    shift: float = Form(3.0, ge=0.0, le=20.0),
    seed: int = Form(-1, ge=-1, le=2**63 - 1),
    megapixels: float = Form(1.0, ge=0.25, le=2.0),
    as_base64: bool = Form(False),
    wait: bool = Form(True),
):
    require_feature("control")
    if canny_low >= canny_high:
        raise HTTPException(400, "canny_low must be smaller than canny_high")
    src = await read_upload(image, megapixels)
    seed = pick_seed(seed)
    w, h = src.size
    work = lambda: execute(lambda name: wf_control(name, prompt, w, h, seed, steps, strength, shift,
                                                   mode, canny_low, canny_high), [src])
    return await dispatch("control", seed, work, wait, as_base64)


@app.get("/", response_class=HTMLResponse)
async def test_page():
    return (ROOT / "test_page.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    features = {name: feature_ready(name) for name in FEATURES}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            stats = (await client.get(f"{COMFY_URL}/system_stats")).json()
    except Exception as e:
        return JSONResponse({"ok": False, "comfyui": f"unreachable: {e}", "features": features}, status_code=503)
    devices = [{"name": d.get("name"), "vram_total_gb": round(d.get("vram_total", 0) / 2**30, 1),
                "vram_free_gb": round(d.get("vram_free", 0) / 2**30, 1)} for d in stats.get("devices", [])]
    return {"ok": True, "auth_required": bool(API_KEY), "features": features,
            "jobs_pending": sum(1 for j in JOBS.values() if j["status"] == "running"), "devices": devices}


@app.on_event("startup")
async def _schedule_warmup():
    # Load the base model before the first real request. Behind Cloudflare a cold
    # first request can exceed the 100 s proxy timeout.
    if os.environ.get("WARMUP", "1") == "1" and feature_ready("generate"):
        asyncio.create_task(_warmup())


async def _warmup():
    started = time.monotonic()
    try:
        await execute(lambda: wf_generate("a simple test image", 256, 256, 0, 1, 1.0, 3.0))
        print(f">> Warm-up done in {time.monotonic() - started:.0f}s, models are loaded", flush=True)
    except Exception as e:
        print(f">> Warm-up failed: {getattr(e, 'detail', e)}", flush=True)


if __name__ == "__main__":
    import uvicorn

    if not API_KEY and HOST not in ("127.0.0.1", "localhost"):
        raise SystemExit("API_KEY is empty but HOST is public. Set API_KEY in .env first.")
    uvicorn.run(app, host=HOST, port=PORT)
