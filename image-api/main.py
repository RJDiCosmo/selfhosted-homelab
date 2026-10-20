"""Image Generation MCP server — wraps ComfyUI's Flux workflow behind an MCP tool."""

import asyncio
import contextvars
import copy
import functools
import inspect
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import docker
import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

COMFYUI_URL = "http://comfyui:8188"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# comfyui idles at ~2.9GB RSS, so it's started/stopped on demand via the Docker
# API (through docker-socket-proxy, not a raw socket mount) instead of running
# continuously.
COMFYUI_CONTAINER = os.environ.get("COMFYUI_CONTAINER", "comfyui")
COMFYUI_IDLE_TIMEOUT = int(os.environ.get("COMFYUI_IDLE_TIMEOUT", "600"))
COMFYUI_READY_TIMEOUT = 180

# Generated PNGs otherwise accumulate in OUTPUT_DIR forever -- nothing else ever
# deletes them (there's no gallery/browse UI, just direct URLs). Age-based cleanup
# keeps disk usage bounded without needing to track which images are still "in use".
IMAGE_RETENTION_DAYS = int(os.environ.get("IMAGE_RETENTION_DAYS", "30"))

docker_client = docker.DockerClient(base_url=os.environ["DOCKER_HOST"])
last_used = time.monotonic()


async def ensure_comfyui_running() -> None:
    global last_used
    last_used = time.monotonic()

    container = docker_client.containers.get(COMFYUI_CONTAINER)
    if container.status != "running":
        container.start()

    deadline = time.monotonic() + COMFYUI_READY_TIMEOUT
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{COMFYUI_URL}/system_stats")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError("ComfyUI did not become ready in time")


async def idle_shutdown_loop() -> None:
    while True:
        await asyncio.sleep(60)
        try:
            container = docker_client.containers.get(COMFYUI_CONTAINER)
            if container.status == "running" and (time.monotonic() - last_used) > COMFYUI_IDLE_TIMEOUT:
                container.stop(timeout=30)
        except docker.errors.DockerException:
            pass


async def cleanup_old_files_loop() -> None:
    cutoff_seconds = IMAGE_RETENTION_DAYS * 86400
    while True:
        now = time.time()
        for path in OUTPUT_DIR.glob("*.png"):
            try:
                if now - path.stat().st_mtime > cutoff_seconds:
                    path.unlink()
            except OSError:
                pass
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(server: FastMCP):
    idle_task = asyncio.create_task(idle_shutdown_loop())
    cleanup_task = asyncio.create_task(cleanup_old_files_loop())
    try:
        yield
    finally:
        idle_task.cancel()
        cleanup_task.cancel()


WORKFLOW_TEMPLATE = json.loads((Path(__file__).parent / "flux_workflow_api.json").read_text())

# Node IDs from flux_workflow_api.json — the KSampler seed is randomized per request
# unless the caller supplies one, so repeat prompts don't return the same image.
POSITIVE_NODE = "4"
NEGATIVE_NODE = "6"
LATENT_NODE = "7"
SAMPLER_NODE = "8"
SAVE_NODE = "10"

mcp = FastMCP(
    name="Image Generation",
    instructions=(
        "Generates images locally via ComfyUI, running a Flux.1-dev GGUF (Q5_K_S) model "
        "on the host GPU. Submit a prompt, get back a URL to the generated PNG. The first "
        "call after idle may take 60-90s while ComfyUI cold-starts. The GPU is shared "
        "with music generation: call status first to check whether it is free. "
        "gpu_status shows VRAM use and what holds the card; stop_model unloads the image "
        "model (musicgen's stop_model unloads the music model). Calls through the remote "
        "gateway answer within ~40s: if generate is still working it returns a job_id "
        "instead of an image_url, so call job_status(job_id) until it reports succeeded."
    ),
    host="0.0.0.0",
    port=8010,
    lifespan=lifespan,
)


class GenerateResult(BaseModel):
    # A plain str, never null (2026-10-19): clients that cached the pre-jobs schema (image_url: required
    # string) reject a null with MCP -32602 and lose the whole reply, job_id included.
    image_url: str = Field("", description="Public URL of the generated image (empty string while the job is still running)")
    pending_image_url: Optional[str] = Field(None, description="While status is running: the URL the image will have once the job finishes (404 until then)")
    job_id: Optional[str] = Field(None, description="Background job id; poll job_status with it while status is running")
    status: str = Field("succeeded", description="succeeded | running")
    message: str = ""


# --- Background jobs (added 2026-10-19) ------------------------------------------------
# Grok Bot cancels any tool call after 60s, and a ComfyUI cold start alone can take that
# long. Calls that arrive through mcp-gateway (it sets X-MCP-Gateway) therefore wait at most
# REMOTE_WAIT_SECONDS: if the work isn't done by then they get a job_id to poll with
# job_status while the work carries on. Local callers (127.0.0.1) still wait for the result.
# Jobs live in memory, so a restart forgets them (finished images are still in OUTPUT_DIR).
REMOTE_WAIT_SECONDS = float(os.environ.get("REMOTE_WAIT_SECONDS", "40"))
JOB_POLL_MAX_SECONDS = 45  # stays under Grok's 60s limit
JOB_KEEP_SECONDS = 24 * 3600
jobs: dict[str, dict] = {}

# Output filename picked when the job starts, so a "running" reply can already name the final URL
# (pending_image_url). Set in background_job before create_task, which copies the context.
_planned_out_name: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("planned_out_name", default=None)
# Public URL prefix for generated PNGs. Must be the browser-reachable path that
# proxies GET /images/{filename} on this server. Trailing slash is optional.
IMAGE_URL_BASE = os.environ.get(
    "IMAGE_URL_BASE", "https://chat.example.com/image-api/images/"
).rstrip("/") + "/"


class JobStatus(BaseModel):
    job_id: str
    tool: str
    status: str = Field(..., description="running | succeeded | failed")
    elapsed_seconds: float
    result: Optional[dict] = Field(None, description="The tool's normal result (e.g. image_url) once succeeded")
    error: Optional[str] = None
    message: str = ""


def _via_gateway(ctx: Optional[Context]) -> bool:
    try:
        request = ctx.request_context.request
    except (AttributeError, ValueError):
        return False
    return request is not None and "x-mcp-gateway" in request.headers


def _job_view(job_id: str) -> JobStatus:
    job = jobs[job_id]
    task = job["task"]
    elapsed = round((job["finished"] or time.monotonic()) - job["started"], 1)
    view = JobStatus(job_id=job_id, tool=job["tool"], status="running", elapsed_seconds=elapsed)
    if not task.done():
        view.message = "Still running; call job_status again with this job_id."
    elif task.cancelled():
        view.status, view.error = "failed", "cancelled (stop_model or a server restart)"
    elif task.exception() is not None:
        view.status, view.error = "failed", f"{type(task.exception()).__name__}: {task.exception()}"
    else:
        view.status, view.result = "succeeded", task.result().model_dump(exclude_none=True)
    return view


def background_job(fn):
    """Tool decorator: runs the call as a job and, for gateway callers, answers after
    REMOTE_WAIT_SECONDS with a job_id when the job isn't finished. Adds a hidden ctx param."""
    sig = inspect.signature(fn)
    result_model = sig.return_annotation

    @functools.wraps(fn)
    async def wrapper(*args, ctx: Optional[Context] = None, **kwargs):
        now = time.monotonic()
        for old in [k for k, j in jobs.items() if j["finished"] and now - j["finished"] > JOB_KEEP_SECONDS]:
            del jobs[old]
        job_id = uuid.uuid4().hex[:12]
        planned = f"{uuid.uuid4().hex}.png"
        token = _planned_out_name.set(planned)
        try:
            task = asyncio.create_task(fn(*args, **kwargs))
        finally:
            _planned_out_name.reset(token)
        jobs[job_id] = {"tool": fn.__name__, "task": task, "started": now, "finished": None}

        def _done(t: asyncio.Task) -> None:
            jobs.get(job_id, {})["finished"] = time.monotonic()
            if not t.cancelled():
                t.exception()  # mark retrieved; job_status reports it

        task.add_done_callback(_done)
        # asyncio.wait never cancels the task, so the job survives a client that hangs up.
        await asyncio.wait({task}, timeout=REMOTE_WAIT_SECONDS if _via_gateway(ctx) else None)
        if task.done():
            result = task.result()  # re-raises a failure as this call's error
            result.job_id = job_id
            return result
        running = result_model(job_id=job_id, status="running", message=(
            f"Still working after {REMOTE_WAIT_SECONDS:.0f}s (a cold start alone can take 60-90s). "
            "Call job_status with this job_id to wait for the result."))
        if "pending_image_url" in result_model.model_fields:
            running.pending_image_url = IMAGE_URL_BASE + planned
            running.message += (" If job_status is unavailable, pending_image_url serves the image once the job "
                                "finishes (404 until then; it never appears if the job fails).")
        return running

    wrapper.__signature__ = sig.replace(parameters=[
        *sig.parameters.values(),
        inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Optional[Context]),
    ])
    wrapper.__annotations__ = {**fn.__annotations__, "ctx": Optional[Context]}
    return wrapper


@mcp.tool()
async def job_status(job_id: str, wait_seconds: int = 30) -> JobStatus:
    """Check a background job started by generate (it returns a job_id when it takes longer
    than ~40s). Waits up to wait_seconds (max 45) for the job to finish before answering, so
    keep calling it until status is succeeded (result holds the image_url) or failed."""
    job = jobs.get(job_id)
    if job is None:
        raise ValueError("Unknown job_id. Jobs are forgotten 24h after finishing or when the server restarts.")
    wait = max(0, min(wait_seconds, JOB_POLL_MAX_SECONDS))
    if wait:
        await asyncio.wait({job["task"]}, timeout=wait)
    return _job_view(job_id)


def build_workflow(
    prompt: str, negative_prompt: str, width: int, height: int, steps: int, seed: Optional[int]
) -> dict:
    wf = copy.deepcopy(WORKFLOW_TEMPLATE)
    wf[POSITIVE_NODE]["inputs"]["text"] = prompt
    wf[NEGATIVE_NODE]["inputs"]["text"] = negative_prompt
    wf[LATENT_NODE]["inputs"]["width"] = width
    wf[LATENT_NODE]["inputs"]["height"] = height
    wf[SAMPLER_NODE]["inputs"]["steps"] = steps
    wf[SAMPLER_NODE]["inputs"]["seed"] = seed if seed is not None else uuid.uuid4().int & 0xFFFFFFFF
    wf[SAVE_NODE]["inputs"]["filename_prefix"] = f"gen_{uuid.uuid4().hex[:8]}"
    return wf


class Status(BaseModel):
    busy: bool = Field(..., description="True if something is generating or holding the GPU right now")
    summary: str = Field(..., description="One sentence: what is running, if anything, and what that means for a new request")


# ACE-Step containers (music-api's on-demand GPU jobs). Only one GPU job fits at a time.
MUSIC_CONTAINERS = {
    "acestep": "the 2B music model",
    "acestep-gradio": "music studio mode (2B)",
    "acestep-xl": "the XL music model",
    "acestep-xl-sft": "the XL sft music model",
}
active_generations = 0


def _running(name: str) -> bool:
    try:
        return docker_client.containers.get(name).status == "running"
    except docker.errors.NotFound:
        return False


@mcp.tool()
async def status() -> Status:
    """Is anything running right now? Short answer on whether the GPU is free for an image.
    Sees image and music generation only, not games or other non-container GPU use."""
    if active_generations:
        return Status(busy=True, summary=(
            f"An image is being generated right now ({active_generations} in progress); "
            "new requests queue behind it."))
    music = [desc for name, desc in MUSIC_CONTAINERS.items() if _running(name)]
    if music:
        return Status(busy=True, summary=(
            f"Music generation is holding the GPU ({', '.join(music)}); an image request now "
            "may fail for lack of GPU memory, so wait a few minutes and check again."))
    if _running(COMFYUI_CONTAINER):
        return Status(busy=False, summary="Idle. ComfyUI is already loaded, so the next image starts right away.")
    return Status(busy=False, summary=(
        "Idle. Nothing is using the GPU; the first image takes an extra 60-90s while ComfyUI starts."))


class GpuProcessInfo(BaseModel):
    name: str
    container: Optional[str] = Field(None, description="Container it runs in; empty for a host program such as a game")
    vram_used_gb: Optional[float] = None


class GpuInfo(BaseModel):
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    gpu_utilization_percent: Optional[int] = None
    temperature_c: Optional[int] = None
    models_loaded: list[str] = Field(default_factory=list, description="Image/music model containers running now, and which server's stop_model unloads them")
    processes: list[GpuProcessInfo] = Field(default_factory=list, description="Model containers and host programs using >= 0.5 GB (e.g. a game)")
    summary: str


STACK_API_URL = os.environ.get("STACK_API_URL", "http://stack-api:8020")
GPU_BIG_GB = 0.5  # host GPU users at or above this are named (games); smaller ones are summed as desktop
GPU_CONTAINERS = {
    COMFYUI_CONTAINER: "Flux image model (imagegen stop_model)",
    **{name: f"{desc} (musicgen stop_model)" for name, desc in MUSIC_CONTAINERS.items()},
}


@mcp.tool()
async def gpu_status() -> GpuInfo:
    """Read-only GPU snapshot: VRAM in use, utilization, temperature, which image/music model
    is loaded (and whose stop_model unloads it), and other GPU users such as games, which
    cannot be stopped from here. Changes nothing."""
    loaded = [f"{name}: {desc}" for name, desc in GPU_CONTAINERS.items() if _running(name)]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{STACK_API_URL}/gpu")
            resp.raise_for_status()
            g = resp.json()
    except httpx.HTTPError:
        return GpuInfo(models_loaded=loaded, summary=(
            "GPU readings unavailable (stack-api is down); "
            + (f"loaded: {'; '.join(loaded)}." if loaded else "no image or music model is loaded.")))
    procs = [GpuProcessInfo(**{k: p.get(k) for k in ("name", "container", "vram_used_gb")}) for p in g["processes"]]
    # The desktop (Xorg, browser, Steam client...) holds a few small slices; only list host
    # programs big enough to matter, such as a game. Unknown usage counts as big.
    big = [p for p in procs if p.container or p.vram_used_gb is None or p.vram_used_gb >= GPU_BIG_GB]
    desktop_gb = round(sum(p.vram_used_gb for p in procs if p not in big), 2)
    procs = big
    others = sorted({p.name for p in procs if not p.container})
    parts = [f"{g['vram_used_gb']} of {g['vram_total_gb']} GB VRAM in use, {g['gpu_utilization_percent']}% busy"]
    parts.append(f"loaded: {'; '.join(loaded)}" if loaded else "no image or music model loaded")
    if others:
        parts.append(f"other GPU programs (not stoppable here): {', '.join(others)}")
    if desktop_gb:
        parts.append(f"desktop apps use {desktop_gb} GB")
    return GpuInfo(
        vram_used_gb=g["vram_used_gb"], vram_total_gb=g["vram_total_gb"],
        gpu_utilization_percent=g["gpu_utilization_percent"], temperature_c=g["temperature_c"],
        models_loaded=loaded, processes=procs, summary="; ".join(parts) + ".")


STOP_GRACE_SECONDS = 10


class StopResult(BaseModel):
    stopped: list[str]
    summary: str


@mcp.tool()
async def stop_model(force: bool = False) -> StopResult:
    """Unload the image model: stops ComfyUI so its VRAM is freed (the next generate cold-starts
    it again, 60-90s). Refuses while an image is being generated unless force=True, which cuts
    that image off. Only touches the image model; musicgen's stop_model unloads music models."""
    if active_generations and not force:
        raise ValueError("An image is being generated right now. Wait for it (job_status), "
                         "or call stop_model with force=true to cut it off.")
    if not _running(COMFYUI_CONTAINER):
        return StopResult(stopped=[], summary="The image model was not loaded; nothing to stop.")
    # ComfyUI ignores SIGTERM, so a 30s grace would just be 30s of waiting before the kill;
    # keep the call well under Grok's 60s tool limit.
    docker_client.containers.get(COMFYUI_CONTAINER).stop(timeout=STOP_GRACE_SECONDS)
    return StopResult(stopped=[COMFYUI_CONTAINER], summary="Stopped ComfyUI; the image model's VRAM is free.")


@mcp.tool()
@background_job
async def generate(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    seed: Optional[int] = None,
) -> GenerateResult:
    """Generate an image from a text prompt. width/height 256-1536, steps 1-50."""
    global active_generations
    active_generations += 1
    try:
        return await _generate(prompt, negative_prompt, width, height, steps, seed)
    finally:
        active_generations -= 1


async def _generate(
    prompt: str, negative_prompt: str, width: int, height: int, steps: int, seed: Optional[int]
) -> GenerateResult:
    width = max(256, min(width, 1536))
    height = max(256, min(height, 1536))
    steps = max(1, min(steps, 50))

    await ensure_comfyui_running()

    client_id = uuid.uuid4().hex
    workflow = build_workflow(prompt, negative_prompt, width, height, steps, seed)

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id}
        )
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]

        image_info = None
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            hist_resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
            hist_resp.raise_for_status()
            hist = hist_resp.json()
            entry = hist.get(prompt_id)
            if entry is not None:
                images = entry.get("outputs", {}).get(SAVE_NODE, {}).get("images", [])
                if images:
                    image_info = images[0]
                break

        if image_info is None:
            raise TimeoutError("Image generation timed out")

        view_resp = await client.get(
            f"{COMFYUI_URL}/view",
            params={
                "filename": image_info["filename"],
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            },
        )
        view_resp.raise_for_status()

        # Unload the model immediately so VRAM is free for the next GPU job
        # (e.g. an ACE-Step generation).
        try:
            await client.post(
                f"{COMFYUI_URL}/free", json={"unload_models": True, "free_memory": True}
            )
        except httpx.HTTPError:
            pass

    out_name = _planned_out_name.get() or f"{uuid.uuid4().hex}.png"
    (OUTPUT_DIR / out_name).write_bytes(view_resp.content)

    return GenerateResult(image_url=IMAGE_URL_BASE + out_name)


# Not an MCP tool: this serves the raw PNG bytes a browser client renders inline,
# so it must stay reachable directly by the browser (unlike MCP traffic, which
# stays on the internal Docker network). Public via the reverse proxy, so serve
# only names this server generates (uuid4 hex + .png).
IMAGE_NAME_RE = re.compile(r"[0-9a-f]{32}\.png")


@mcp.custom_route("/images/{filename}", methods=["GET"])
async def get_image(request: Request):
    filename = request.path_params["filename"]
    if not IMAGE_NAME_RE.fullmatch(filename):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.exists():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
