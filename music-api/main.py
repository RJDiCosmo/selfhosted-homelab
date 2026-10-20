"""Music Generation MCP server — wraps ACE-Step 1.5's REST API behind an MCP tool.

Same on-demand GPU container pattern as image-api: the acestep
container is started/stopped on demand via docker-socket-proxy rather than run
continuously, since it's used rarely and would otherwise idle-occupy VRAM.
"""

import asyncio
import contextvars
import functools
import inspect
import json
import os
import re
import shutil
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

ACESTEP_URL = "http://acestep:8001"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read-only bind mount of source tracks so modify_song can reference them by
# filename without needing an HTTP upload endpoint.
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/uploads"))

# acestep only runs on demand — started/stopped via the Docker API (through
# docker-socket-proxy, shared with image-api's comfyui controller) rather than
# continuously, since it's used rarely and would otherwise idle-occupy VRAM.
ACESTEP_CONTAINER = os.environ.get("ACESTEP_CONTAINER", "acestep")
ACESTEP_IDLE_TIMEOUT = int(os.environ.get("MUSIC_IDLE_TIMEOUT", "600"))
# Cold start loads the DiT (+ optional LM) checkpoints, which is slower than
# ComfyUI's cold start -- upstream's own healthcheck uses a 300s start_period.
ACESTEP_READY_TIMEOUT = 300

# Generated MP3s otherwise accumulate in OUTPUT_DIR forever -- nothing else ever
# deletes them. Age-based cleanup keeps disk usage bounded, same pattern as
# image-api's IMAGE_RETENTION_DAYS.
MUSIC_RETENTION_DAYS = int(os.environ.get("MUSIC_RETENTION_DAYS", "30"))

docker_client = docker.DockerClient(base_url=os.environ["DOCKER_HOST"])

# --- Studio mode -------------------------------------------------------------
# A second, mutually exclusive way to run ACE-Step: instead of the on-demand
# REST `acestep` container (idle-timeout, driven by the tools below), studio mode
# starts `acestep-gradio` (same 2B turbo DiT, Gradio server, kept resident in
# VRAM) plus `studio-ui` (fspecii/ace-step-ui, served at STUDIO_PUBLIC_URL behind
# the edge proxy) and keeps them up until studio_off. Both acestep variants
# load the same model into the same 12GB card, so only one may run at a time.
GRADIO_CONTAINER = os.environ.get("ACESTEP_GRADIO_CONTAINER", "acestep-gradio")
STUDIO_UI_CONTAINER = os.environ.get("STUDIO_UI_CONTAINER", "studio-ui")
COMFYUI_CONTAINER = os.environ.get("COMFYUI_CONTAINER", "comfyui")
GRADIO_URL = "http://acestep-gradio:7860"
STUDIO_UI_URL = "http://studio-ui:3000"
# Browser origin of the studio UI, no trailing slash. Audio paths from the UI are appended to it.
STUDIO_PUBLIC_URL = os.environ.get("STUDIO_PUBLIC_URL", "https://studio.example.com").rstrip("/")
# Username sent to the studio UI's passwordless /api/auth/setup. Not a password.
STUDIO_UI_USERNAME = os.environ.get("STUDIO_UI_USERNAME", "studio")
last_used = time.monotonic()

# --- XL pathway (added 2026-09-26) ---------------------------------------------
# The 4B XL turbo DiT, int8-quantized under Gradio (it OOMs at full bf16, and the
# REST server can't quantize). generate_song_xl starts acestep-xl + studio-ui on
# demand, drives the UI backend the same way studio_generate does, copies the result
# into OUTPUT_DIR, and the idle loop stops both after XL_IDLE_TIMEOUT. acestep-xl
# carries the network alias "acestep-gradio", so the UI and GRADIO_URL reach it
# unchanged. Only one of acestep / acestep-gradio / acestep-xl may run at a time.
XL_CONTAINER = os.environ.get("ACESTEP_XL_CONTAINER", "acestep-xl")
# Non-distilled XL (50 steps, working CFG). Same image/args as acestep-xl with
# ACESTEP_CONFIG_PATH=acestep-v15-xl-sft; carries the same "acestep-gradio" alias and
# never runs together with acestep-xl (generate_song_xl(model="sft") swaps them).
XL_SFT_CONTAINER = os.environ.get("ACESTEP_XL_SFT_CONTAINER", "acestep-xl-sft")
XL_IDLE_TIMEOUT = int(os.environ.get("XL_IDLE_TIMEOUT", "900"))
XL_THINKING_DEFAULT = os.environ.get("XL_THINKING_DEFAULT", "0") == "1"  # LM planner on for XL (added 2026-10-17)
XL_COT_CAPTION = os.environ.get("XL_COT_CAPTION", "0") == "1"  # let the LM rewrite our caption (it invented "vinyl crackle" 2026-10-17, so off)
XL_READY_TIMEOUT = 420  # measured cold start ~130s (20GB fp32 read + int8 quantize)
xl_last_used = time.monotonic()
# Studio mode can also run XL (studio_on(model="xl")): same acestep-xl container, but
# resident with no idle stop. The mode lives in a flag file rather than in memory, so a
# music-api restart doesn't mistake a studio-XL session for on-demand XL and idle-stop it.
STUDIO_XL_FLAG = OUTPUT_DIR / ".studio_xl"
# studio-ui sends Enable DCW=True to ACE-Step (right for 8-step turbo, wrecks the 50-step sft; found
# 2026-10-11). It reads this flag file (mounted at /flags in both containers): present = DCW off.
DCW_OFF_FLAG = Path("/flags/dcw_off")


def _set_dcw(enabled: bool) -> None:
    try:
        if enabled:
            DCW_OFF_FLAG.unlink(missing_ok=True)
        else:
            DCW_OFF_FLAG.touch()
    except OSError as e:
        if not enabled:  # sft without the switch would produce noise; fail loudly
            raise ValueError(f"cannot set the DCW-off flag at {DCW_OFF_FLAG}: {e}")

# Serializes generations across concurrent tool calls so overlapping requests don't
# race on starting/stopping the ACE-Step containers that share the one GPU.
generation_lock = asyncio.Lock()


async def ensure_acestep_running() -> None:
    global last_used
    last_used = time.monotonic()

    container = docker_client.containers.get(ACESTEP_CONTAINER)
    if container.status != "running":
        # Checked immediately before start() (no await in between) so the REST
        # acestep can never be started while studio's Gradio copy holds the GPU.
        _refuse_if_studio()
        container.start()

    deadline = time.monotonic() + ACESTEP_READY_TIMEOUT
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{ACESTEP_URL}/health")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3)
    raise TimeoutError("ACE-Step did not become ready in time")


async def idle_shutdown_loop() -> None:
    while True:
        await asyncio.sleep(60)
        try:
            container = docker_client.containers.get(ACESTEP_CONTAINER)
            if container.status == "running" and (time.monotonic() - last_used) > ACESTEP_IDLE_TIMEOUT:
                container.stop(timeout=30)
        except docker.errors.DockerException:
            pass
        # XL pathway: skip while a generation holds the lock (a long XL job must not
        # be stopped mid-run just because it started more than XL_IDLE_TIMEOUT ago).
        try:
            if (_xl_active() and not STUDIO_XL_FLAG.exists() and not generation_lock.locked()
                    and (time.monotonic() - xl_last_used) > XL_IDLE_TIMEOUT):
                await _stop_xl()
        except docker.errors.DockerException:
            pass


async def cleanup_old_files_loop() -> None:
    cutoff_seconds = MUSIC_RETENTION_DAYS * 86400
    while True:
        now = time.time()
        for path in OUTPUT_DIR.glob("*.mp3"):
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


mcp = FastMCP(
    name="Music Generation",
    instructions=(
        "Generates and modifies music locally via ACE-Step 1.5 (2B turbo DiT model) "
        "running on the host GPU. generate_quick makes short instrumental clips/loops; "
        "generate_song makes longer pieces with sung lyrics. list_tracks shows source "
        "tracks placed in the uploads directory; modify_song reworks one of them via cover (regenerate guided by a "
        "new prompt/lyrics) or repaint (regenerate just one section). All tools return "
        "a URL to the resulting audio file. The first call after idle may take a couple "
        "minutes while ACE-Step cold-starts and loads its model checkpoints. "
        "Three pathways: generate_song/generate_quick/modify_song use the 2B model (REST, "
        "on demand); generate_song_xl uses the 4B XL model (on demand, idle-stopped); "
        f"studio_on/studio_off toggle the resident 2B web UI at {STUDIO_PUBLIC_URL}. "
        "The GPU is shared with image generation: call status first to check whether it is free. "
        "gpu_status shows VRAM use and what holds the card; stop_model unloads every music model "
        "(imagegen's stop_model unloads the image model). Calls through the remote gateway answer "
        "within ~40s: if a generate/modify/describe call is still working it returns a job_id "
        "instead of the result, so call job_status(job_id) until it reports succeeded. Through the "
        "gateway, call studio_on and studio_generate with wait=false and poll studio_status/studio_job."
    ),
    host="0.0.0.0",
    port=8030,
    lifespan=lifespan,
)


class MusicResult(BaseModel):
    # A plain str, never null (2026-10-19): clients that cached the pre-jobs schema (audio_url: required
    # string) reject a null with MCP -32602 and lose the whole reply, job_id included.
    audio_url: str = Field("", description="Public URL of the generated audio file (empty string while the job is still running)")
    pending_audio_url: Optional[str] = Field(None, description="While status is running: the URL the file will have once the job finishes (404 until then)")
    job_id: Optional[str] = Field(None, description="Background job id; poll job_status with it while status is running")
    status: str = Field("succeeded", description="succeeded | running")
    message: str = ""


# --- Background jobs (added 2026-10-19) ------------------------------------------------
# Grok Bot cancels any tool call after 60s, and a cold start or a long track takes longer
# than that. Calls that arrive through mcp-gateway (it sets X-MCP-Gateway) therefore wait at most
# REMOTE_WAIT_SECONDS: if the work isn't done by then they get a job_id to poll with
# job_status while the work carries on. Local callers (127.0.0.1) still wait for the result.
# Jobs live in memory, so a restart forgets them (finished tracks are still in OUTPUT_DIR).
REMOTE_WAIT_SECONDS = float(os.environ.get("REMOTE_WAIT_SECONDS", "40"))
JOB_POLL_MAX_SECONDS = 45  # stays under Grok's 60s limit
JOB_KEEP_SECONDS = 24 * 3600
jobs: dict[str, dict] = {}
# Output filename picked when the job starts, so a "running" reply can already name the final URL
# (pending_audio_url). Set in background_job before create_task, which copies the context.
_planned_out_name: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("planned_out_name", default=None)
# Public URL prefix for generated MP3s. Must be the browser-reachable path that
# proxies GET /audio/{filename} on this server. Trailing slash is optional.
AUDIO_URL_BASE = os.environ.get(
    "AUDIO_URL_BASE", "https://chat.example.com/music-api/audio/"
).rstrip("/") + "/"


class JobStatus(BaseModel):
    job_id: str
    tool: str
    status: str = Field(..., description="running | succeeded | failed")
    elapsed_seconds: float
    result: Optional[dict] = Field(None, description="The tool's normal result (e.g. audio_url) once succeeded")
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
        view.status, view.error = "failed", "cancelled (server restart)"
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
        planned = f"{uuid.uuid4().hex}.mp3"
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
            f"Still working after {REMOTE_WAIT_SECONDS:.0f}s (cold starts take 1-3 minutes, XL tracks longer). "
            "Call job_status with this job_id to wait for the result."))
        if "pending_audio_url" in result_model.model_fields:
            running.pending_audio_url = AUDIO_URL_BASE + planned
            running.message += (" If job_status is unavailable, pending_audio_url serves the file once the job "
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
    """Check a background job started by generate_song, generate_song_xl, generate_quick,
    modify_song or describe_track (they return a job_id when they take longer than ~40s).
    Waits up to wait_seconds (max 45) for the job to finish before answering, so keep calling
    it until status is succeeded (result holds the audio_url or description) or failed."""
    job = jobs.get(job_id)
    if job is None:
        raise ValueError("Unknown job_id. Jobs are forgotten 24h after finishing or when the server restarts.")
    wait = max(0, min(wait_seconds, JOB_POLL_MAX_SECONDS))
    if wait:
        await asyncio.wait({job["task"]}, timeout=wait)
    return _job_view(job_id)


class TrackList(BaseModel):
    filenames: list[str]


async def _submit_and_wait(
    client: httpx.AsyncClient,
    data: dict,
    files: Optional[dict] = None,
) -> bytes:
    """POST to /release_task (JSON if files is None, multipart otherwise), poll
    /query_result until done, and return the resulting audio bytes."""
    if files is None:
        resp = await client.post(f"{ACESTEP_URL}/release_task", json=data)
    else:
        resp = await client.post(f"{ACESTEP_URL}/release_task", data=data, files=files)
    resp.raise_for_status()
    envelope = resp.json()
    if envelope.get("code") != 200:
        raise ValueError(f"ACE-Step rejected task: {envelope.get('error')}")
    task_id = envelope["data"]["task_id"]

    file_path = None
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        await asyncio.sleep(3)
        query_resp = await client.post(
            f"{ACESTEP_URL}/query_result", json={"task_id_list": [task_id]}
        )
        query_resp.raise_for_status()
        entries = query_resp.json().get("data", [])
        entry = next((e for e in entries if e.get("task_id") == task_id), None)
        if entry is None:
            continue
        if entry["status"] == 2:
            raise ValueError(f"ACE-Step generation failed for task {task_id}")
        if entry["status"] == 1:
            results = json.loads(entry["result"])
            file_path = results[0]["file"]
            break

    if file_path is None:
        raise TimeoutError("Music generation timed out")

    audio_resp = await client.get(f"{ACESTEP_URL}{file_path}", timeout=60.0)
    audio_resp.raise_for_status()
    return audio_resp.content


async def _run_generation(data: dict, files: Optional[dict] = None) -> MusicResult:
    _refuse_if_studio()
    if _xl_active():
        # XL is ours too (on-demand, idle-timeout), so switch automatically rather than
        # refuse. Under the lock so an in-flight XL generation finishes first.
        async with generation_lock:
            await _stop_xl()
    await ensure_acestep_running()

    async with generation_lock:
        # Re-check under the lock: studio_on may have run while we waited for acestep.
        _refuse_if_studio()
        # describe_track stops acestep when it finishes (to drop its LM), so a call that
        # queued behind it would otherwise submit to a stopped container.
        await ensure_acestep_running()
        async with httpx.AsyncClient(timeout=30.0) as client:
            audio_bytes = await _submit_and_wait(client, data, files)

    out_name = _planned_out_name.get() or f"{uuid.uuid4().hex}.mp3"
    (OUTPUT_DIR / out_name).write_bytes(audio_bytes)

    return MusicResult(audio_url=AUDIO_URL_BASE + out_name)


async def _generate(
    prompt: str,
    lyrics: str,
    duration: int,
    steps: int,
    seed: Optional[int],
    bpm: Optional[int] = None,
    infer_method: str = "ode",
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    guidance_scale: float = 7.0,
    vocal_language: Optional[str] = None,
    key_scale: Optional[str] = None,
) -> MusicResult:
    task_body = {
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_format": "mp3",
        "audio_duration": duration,
        "inference_steps": steps,
        "infer_method": infer_method,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "guidance_scale": guidance_scale,
    }
    if seed is not None:
        # ACE-Step's use_random_seed defaults to True and silently overrides `seed`
        # (found 2026-09-26: every REST seed before this was a random roll).
        task_body["seed"] = seed
        task_body["use_random_seed"] = False
    if bpm is not None:
        task_body["bpm"] = bpm
    if vocal_language is not None:
        task_body["vocal_language"] = vocal_language
    # Dedicated keyscale field (see _resolve_key_scale): explicit arg, else the prompt's "in <key>" tag.
    resolved_key = _resolve_key_scale(key_scale, prompt)
    if resolved_key:
        task_body["key_scale"] = resolved_key
    return await _run_generation(task_body)


@mcp.tool()
@background_job
async def generate_quick(
    prompt: str,
    duration: int = 20,
    seed: Optional[int] = None,
    bpm: Optional[int] = None,
    infer_method: str = "ode",
) -> MusicResult:
    """Generate a short instrumental music clip/loop. duration in seconds, 5-30.
    bpm sets tempo via ACE-Step's own dedicated field (more reliable than a bpm
    tag in the prompt text alone). infer_method is 'ode' (default) or 'sde' --
    'sde' is worth trying if a track has rhythm/tempo instability."""
    duration = max(5, min(duration, 30))
    return await _generate(
        prompt, lyrics="", duration=duration, steps=8, seed=seed,
        bpm=bpm, infer_method=infer_method,
    )


@mcp.tool()
@background_job
async def generate_song(
    prompt: str,
    lyrics: str = "",
    duration: int = 90,
    seed: Optional[int] = None,
    bpm: Optional[int] = None,
    infer_method: str = "ode",
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    guidance_scale: float = 7.0,
    vocal_language: Optional[str] = None,
    key_scale: Optional[str] = None,
) -> MusicResult:
    """Generate a longer piece of music, optionally with sung lyrics. duration in
    seconds, 30-240.
    key_scale ("F major", "D minor") is sent as ACE-Step's dedicated keyscale field; if
    omitted it is taken from an "in <Note> major|minor" tag in the prompt.
    bpm sets tempo via ACE-Step's own dedicated field (more reliable than a bpm
    tag in the prompt text alone -- the model treats a text bpm tag as a soft
    hint, not a hard constraint).
    infer_method is 'ode' (default, faster) or 'sde' -- 'sde' is worth trying if
    a track shows rhythm/tempo instability at section transitions, per ACE-Step's
    own reported (unresolved) rhythm-instability issue on dance/EDM tracks over
    ~3 minutes (github.com/ace-step/ACE-Step-1.5 issue #1203).
    cfg_interval_start/cfg_interval_end (0.0-1.0) restrict classifier-free guidance
    to a sub-range of the diffusion timeline -- narrowing this window is another
    knob worth trying against the same instability if bpm/infer_method alone
    doesn't fix it. Leave at the defaults (0.0, 1.0 = full range) otherwise.
    guidance_scale (default 7.0, typical range 5.0-9.0) controls how strictly the
    output follows the prompt tags -- raise it (e.g. 8.5-9.0) if vocal delivery or
    mood tags (e.g. "fierce intensity", "raw conviction") aren't coming through
    strongly enough; lower it for looser/more creative interpretation.
    vocal_language is an ISO 639-1 code (default 'en' if omitted) telling ACE-Step
    what language to sing the lyrics in -- write the `lyrics` text itself in that
    language too, this only sets the model's expected vocal-language mode. Supported
    codes include 'en', 'hi' (Hindi), 'ta' (Tamil), 'pa' (Punjabi), 'ar' (Arabic),
    'es', 'fr', 'de', 'ja', 'ko', 'zh', and more (see ACE-Step's own constants.py
    VALID_LANGUAGES list for the full set).
    Note: ACE-Step's 'shift' (timestep shift) parameter is NOT exposed here --
    per ACE-Step's own docs it only affects base models, not the turbo model
    this stack runs, so it would have no effect."""
    duration = max(30, min(duration, 240))
    return await _generate(
        prompt, lyrics=lyrics, duration=duration, steps=8, seed=seed,
        bpm=bpm, infer_method=infer_method,
        cfg_interval_start=cfg_interval_start, cfg_interval_end=cfg_interval_end,
        guidance_scale=guidance_scale, vocal_language=vocal_language,
        key_scale=key_scale,
    )


@mcp.tool()
async def list_tracks() -> TrackList:
    """List uploaded source tracks available for modify_song. Files placed in the
    uploads directory show up here by filename."""
    names = sorted(p.name for p in UPLOAD_DIR.glob("*") if p.is_file())
    return TrackList(filenames=names)


@mcp.tool()
@background_job
async def modify_song(
    filename: str,
    prompt: str,
    task_type: str = "cover",
    lyrics: str = "",
    duration: int = 90,
    cover_noise_strength: float = 0.5,
    repainting_start: float = 0.0,
    repainting_end: Optional[float] = None,
    seed: Optional[int] = None,
    bpm: Optional[int] = None,
    infer_method: str = "ode",
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    guidance_scale: float = 7.0,
    vocal_language: Optional[str] = None,
) -> MusicResult:
    """Modify an uploaded source track (see list_tracks for available filenames).
    task_type: 'cover' regenerates the whole track guided by prompt/lyrics, staying
    closer to the source the higher cover_noise_strength is (0=totally new, 1=very
    close to source). 'repaint' regenerates only the [repainting_start, repainting_end]
    section (seconds) and leaves the rest of the track untouched -- leave
    repainting_end unset to repaint through the end of the track.
    bpm/infer_method/cfg_interval_start/cfg_interval_end/guidance_scale/vocal_language:
    same tempo/rhythm-stability/prompt-adherence/language knobs as generate_song --
    see that tool's docstring for details."""
    if task_type not in ("cover", "repaint"):
        raise ValueError("task_type must be 'cover' or 'repaint'")

    path = (UPLOAD_DIR / filename).resolve()
    if path.parent != UPLOAD_DIR.resolve() or not path.exists():
        raise ValueError(f"Track '{filename}' not found -- call list_tracks to see what's available")

    duration = max(30, min(duration, 240))
    data = {
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_format": "mp3",
        "audio_duration": str(duration),
        "inference_steps": "8",
        "task_type": task_type,
        "infer_method": infer_method,
        "cfg_interval_start": str(cfg_interval_start),
        "cfg_interval_end": str(cfg_interval_end),
        "guidance_scale": str(guidance_scale),
    }
    if task_type == "cover":
        data["cover_noise_strength"] = str(cover_noise_strength)
    else:
        data["repainting_start"] = str(repainting_start)
        if repainting_end is not None:
            data["repainting_end"] = str(repainting_end)
    if seed is not None:
        data["seed"] = str(seed)
        data["use_random_seed"] = "false"
    if bpm is not None:
        data["bpm"] = str(bpm)
    if vocal_language is not None:
        data["vocal_language"] = vocal_language

    files = {"src_audio": (filename, path.read_bytes(), "audio/mpeg")}
    return await _run_generation(data, files)


INBOX_DIR = Path(os.environ.get("INBOX_DIR", "/inbox"))
DESCRIBE_LM = os.environ.get("DESCRIBE_LM", "acestep-5Hz-lm-1.7B")
DESCRIBE_TIMEOUT = float(os.environ.get("DESCRIBE_TIMEOUT", "600"))


class TrackDescription(BaseModel):
    source: str = ""
    job_id: Optional[str] = Field(None, description="Background job id; poll job_status with it while status is running")
    status: str = Field("succeeded", description="succeeded | running")
    message: str = ""
    caption: str = ""
    genre: str = ""
    bpm: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    language: str = ""
    is_instrumental: Optional[bool] = None
    duration: Optional[float] = None
    lyrics: str = ""
    seconds_taken: float = 0.0


def _resolve_describe_source(source: str) -> Path:
    """A filename in /uploads, or a music-api take given as hash, hash.mp3 or its audio_url."""
    name = source.strip().rsplit("/", 1)[-1]
    up = (UPLOAD_DIR / name).resolve()
    if up.parent == UPLOAD_DIR.resolve() and up.is_file():
        return up
    stem = name[:-4] if name.endswith(".mp3") else name
    out = (OUTPUT_DIR / f"{stem}.mp3").resolve()
    if out.parent == OUTPUT_DIR.resolve() and out.is_file():
        return out
    raise ValueError(f"'{source}' not found in uploads (see list_tracks) or as a generated take")


async def _acestep_api(client: httpx.AsyncClient, path: str, body: dict) -> dict:
    resp = await client.post(f"{ACESTEP_URL}{path}", json=body)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error") or (payload.get("code") not in (None, 200)):
        raise RuntimeError(f"ACE-Step {path} failed: {payload.get('error') or payload}")
    return payload.get("data") or {}


@mcp.tool()
@background_job
async def describe_track(source: str, transcribe_lyrics: bool = True) -> TrackDescription:
    """Listen to an existing song with ACE-Step's own audio-understanding LM and return
    its caption (genre/instruments/mood, in ACE-Step's tag vocabulary), BPM, key, time
    signature, language and a rough lyric transcription -- a starting point for a
    generate_song prompt. source = a filename in uploads (list_tracks) or a generated
    take's hash/audio_url. Uses the REST 2B container: loads the 1.7B LM on demand, then
    stops the container afterwards so normal generation is unaffected. Takes a few
    minutes; refused while studio mode is on."""
    path = _resolve_describe_source(source)
    started = time.monotonic()
    _refuse_if_studio()
    async with generation_lock:
        _refuse_if_studio()
        await _stop_xl()
        await ensure_acestep_running()
        job = uuid.uuid4().hex
        job_dir = INBOX_DIR / job
        job_dir.mkdir(parents=True)
        try:
            shutil.copyfile(path, job_dir / path.name)
            async with httpx.AsyncClient(timeout=DESCRIBE_TIMEOUT) as client:
                await _acestep_api(client, "/v1/init", {
                    "model": "acestep-v15-turbo", "init_llm": True, "lm_model_path": DESCRIBE_LM})
                await _acestep_api(client, "/v1/dataset/scan", {
                    "audio_dir": f"/app/inbox/{job}", "dataset_name": f"describe-{job}",
                    "all_instrumental": False})
                # No save_path: the label JSON lands in the job dir, deleted below.
                data = await _acestep_api(client, "/v1/dataset/auto_label", {
                    "transcribe_lyrics": transcribe_lyrics})
            samples = data.get("samples") or []
            if not samples:
                raise RuntimeError(f"ACE-Step returned no labels: {data.get('message')}")
            s0 = samples[0]
            try:
                bpm = int(float(s0.get("bpm"))) or None
            except (TypeError, ValueError):
                bpm = None
            return TrackDescription(
                source=path.name, caption=s0.get("caption") or "", genre=s0.get("genre") or "",
                bpm=bpm,
                keyscale=s0.get("keyscale") or "", timesignature=str(s0.get("timesignature") or ""),
                language=s0.get("language") or "", is_instrumental=s0.get("is_instrumental"),
                duration=s0.get("duration"), lyrics=s0.get("lyrics") or "",
                seconds_taken=round(time.monotonic() - started, 1),
            )
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            # Drop the LM: a fresh acestep start comes back with ACESTEP_INIT_LLM=false.
            try:
                c = docker_client.containers.get(ACESTEP_CONTAINER)
                if c.status == "running":
                    c.stop(timeout=30)
            except docker.errors.DockerException:
                pass


# Not an MCP tool: serves the raw audio bytes a browser client plays inline, so
# it must stay reachable directly by the browser (unlike MCP traffic, which stays
# on the internal Docker network).
def _container_status(name: str) -> str:
    try:
        return docker_client.containers.get(name).status
    except docker.errors.NotFound:
        return "missing"


def _xl_active() -> bool:
    return (_container_status(XL_CONTAINER) == "running"
            or _container_status(XL_SFT_CONTAINER) == "running")


def _studio_xl_on() -> bool:
    return STUDIO_XL_FLAG.exists() and _xl_active()


def _studio_active() -> bool:
    # Studio = the resident 2B Gradio, or acestep-xl started by studio_on(model="xl").
    return _container_status(GRADIO_CONTAINER) == "running" or _studio_xl_on()


async def _stop_xl() -> None:
    """Stop acestep-xl and the UI it was using. Callers hold generation_lock, or accept
    that an XL job in flight would be cut off."""
    for name in (STUDIO_UI_CONTAINER, XL_CONTAINER, XL_SFT_CONTAINER):
        try:
            c = docker_client.containers.get(name)
            if c.status == "running":
                c.stop(timeout=30)
        except docker.errors.NotFound:
            pass


def _refuse_if_studio() -> None:
    if _studio_active():
        raise ValueError(
            "Studio mode is on (ACE-Step is running in the studio Gradio server, "
            "not the REST API these tools use). Call studio_off first, then retry."
        )


async def _wait_http(url: str, timeout: float, ok_below: int = 500) -> bool:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(url)).status_code < ok_below:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3)
    return False


class StudioStatus(BaseModel):
    studio_on: bool = Field(..., description="True when the Gradio server and UI are both running")
    gradio: str = Field(..., description="acestep-gradio container status")
    ui: str = Field(..., description="studio-ui container status")
    rest_acestep: str = Field(..., description="on-demand REST acestep (2B) container status")
    xl: str = Field("", description="acestep-xl (4B XL turbo) container status")
    xl_sft: str = Field("", description="acestep-xl-sft (4B XL non-distilled) container status")
    model: str = Field("", description="Model studio is running: '2b', 'xl', or '' when studio is off")
    url: str = Field(..., description="Studio UI URL (Basic Auth required)")
    note: str = ""


def _studio_status(note: str = "") -> StudioStatus:
    g, u = _container_status(GRADIO_CONTAINER), _container_status(STUDIO_UI_CONTAINER)
    model = "2b" if g == "running" else ("xl" if _studio_xl_on() else "")
    return StudioStatus(
        studio_on=(bool(model) and u == "running"),
        model=model,
        gradio=g, ui=u,
        rest_acestep=_container_status(ACESTEP_CONTAINER),
        xl=_container_status(XL_CONTAINER),
        xl_sft=_container_status(XL_SFT_CONTAINER),
        url=STUDIO_PUBLIC_URL, note=note,
    )


async def _studio_on(wait: bool = True, model: str = "2b") -> StudioStatus:
    model = model.strip().lower()
    if model not in ("2b", "xl"):
        raise ValueError("model must be '2b' or 'xl'")
    backend = XL_CONTAINER if model == "xl" else GRADIO_CONTAINER
    other = GRADIO_CONTAINER if model == "xl" else XL_CONTAINER
    for name in (backend, STUDIO_UI_CONTAINER):
        if _container_status(name) == "missing":
            raise ValueError(
                f"Container {name} does not exist. Create the {name} container "
                "(and studio-ui) before calling this tool."
            )
    if _container_status(COMFYUI_CONTAINER) == "running":
        raise ValueError(
            "comfyui is running (image generation holds the GPU). Wait for it to idle "
            "out or stop it, then call studio_on again."
        )
    # Serialize against in-flight generations so we never stop a container mid-task.
    async with generation_lock:
        rest = docker_client.containers.get(ACESTEP_CONTAINER)
        if rest.status == "running":
            rest.stop(timeout=30)
        # Both Gradio copies answer as "acestep-gradio"; the other one must be gone
        # before this one starts, or the UI could reach either.
        if _container_status(other) == "running":
            docker_client.containers.get(other).stop(timeout=30)
        # Set/clear the studio-XL flag BEFORE starting, so the idle loop never treats
        # a starting studio-XL as on-demand XL. An already-running on-demand XL is
        # simply adopted into studio mode.
        if model == "xl":
            STUDIO_XL_FLAG.touch()
        else:
            STUDIO_XL_FLAG.unlink(missing_ok=True)
        if _container_status(backend) != "running":
            docker_client.containers.get(backend).start()
    if not wait:
        return _studio_status("starting; poll studio_status")
    # On a startup timeout, roll back to "off" rather than leaving a half-started
    # studio holding the GPU (which would also make the REST tools refuse).
    ready_timeout = XL_READY_TIMEOUT if model == "xl" else ACESTEP_READY_TIMEOUT
    if not await _wait_http(f"{GRADIO_URL}/gradio_api/info", ready_timeout):
        await _studio_off()
        raise TimeoutError(f"{backend} did not become ready in time; studio rolled back to off (check stack_logs {backend})")
    docker_client.containers.get(STUDIO_UI_CONTAINER).start()
    if not await _wait_http(f"{STUDIO_UI_URL}/api/generate/health", 90):
        await _studio_off()
        raise TimeoutError("studio-ui did not become ready in time; studio rolled back to off (check stack_logs studio-ui)")
    return _studio_status("ready")


async def _studio_off() -> StudioStatus:
    # Also stops acestep-xl: "off" means the GPU is free, whichever Gradio copy was up.
    for name in (STUDIO_UI_CONTAINER, GRADIO_CONTAINER, XL_CONTAINER, XL_SFT_CONTAINER):
        try:
            c = docker_client.containers.get(name)
            if c.status == "running":
                c.stop(timeout=30)
        except docker.errors.NotFound:
            pass
    STUDIO_XL_FLAG.unlink(missing_ok=True)
    return _studio_status("off; the on-demand REST tools work again")


@mcp.tool()
async def studio_on(wait: bool = True, model: str = "2b") -> StudioStatus:
    """Turn studio mode ON: stops the REST acestep / on-demand XL, starts a resident Gradio
    ACE-Step server plus the studio web UI, kept up until studio_off.
    model="2b" (default, ~1 min start) or "xl" (4B XL int8, ~2-3 min start, ~6GB VRAM).
    Calling it while studio is on with the other model switches models. musicgen generate
    tools are refused while it is on. wait=False returns immediately; poll studio_status."""
    return await _studio_on(wait, model)


@mcp.tool()
async def studio_off() -> StudioStatus:
    """Turn studio mode OFF: stops the UI and Gradio server (and acestep-xl if it is
    up), freeing the GPU. The on-demand REST generate tools work again afterwards."""
    return await _studio_off()


@mcp.tool()
async def studio_status() -> StudioStatus:
    """Report whether studio mode is on, plus container states."""
    return _studio_status()


class Status(BaseModel):
    busy: bool = Field(..., description="True if something is generating or holding the GPU right now")
    summary: str = Field(..., description="One sentence: what is running, if anything, and what that means for a new request")


@mcp.tool()
async def status() -> Status:
    """Is anything running right now? Short answer on whether the GPU is free for music.
    Sees music and image generation only, not games or other non-container GPU use."""
    if generation_lock.locked():
        return Status(busy=True, summary=(
            "A music job is in progress; new requests queue behind it and start when it finishes."))
    if _container_status(COMFYUI_CONTAINER) == "running":
        return Status(busy=True, summary=(
            "Image generation (ComfyUI) is holding the GPU; a music request now may fail for "
            "lack of GPU memory, so wait a few minutes and check again."))
    s = _studio_status()
    if s.model:
        return Status(busy=True, summary=(
            f"Studio mode is on ({s.model}); generate_song and generate_song_xl refuse until "
            "studio_off, but studio_generate works."))
    if _xl_active():
        return Status(busy=False, summary=(
            "Idle. The XL model is loaded, so generate_song_xl starts right away."))
    if s.rest_acestep == "running":
        return Status(busy=False, summary=(
            "Idle. The 2B model is loaded, so generate_song/generate_quick start right away."))
    return Status(busy=False, summary=(
        "Idle. Nothing is using the GPU; the first call cold-starts the model "
        "(about 1-3 minutes, longer for XL)."))


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
# Every container that loads a model onto the GPU (studio-ui doesn't, but stop_model stops it too).
MUSIC_GPU_CONTAINERS = {
    ACESTEP_CONTAINER: "2B music model",
    GRADIO_CONTAINER: "2B music model, studio mode",
    XL_CONTAINER: "XL turbo music model",
    XL_SFT_CONTAINER: "XL sft music model",
}


@mcp.tool()
async def gpu_status() -> GpuInfo:
    """Read-only GPU snapshot: VRAM in use, utilization, temperature, which music/image model
    is loaded (and whose stop_model unloads it), and other GPU users such as games, which
    cannot be stopped from here. Changes nothing."""
    loaded = [f"{name}: {desc} (musicgen stop_model)" for name, desc in MUSIC_GPU_CONTAINERS.items()
              if _container_status(name) == "running"]
    if _container_status(COMFYUI_CONTAINER) == "running":
        loaded.append(f"{COMFYUI_CONTAINER}: Flux image model (imagegen stop_model)")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{STACK_API_URL}/gpu")
            resp.raise_for_status()
            g = resp.json()
    except httpx.HTTPError:
        return GpuInfo(models_loaded=loaded, summary=(
            "GPU readings unavailable (stack-api is down); "
            + (f"loaded: {'; '.join(loaded)}." if loaded else "no music or image model is loaded.")))
    procs = [GpuProcessInfo(**{k: p.get(k) for k in ("name", "container", "vram_used_gb")}) for p in g["processes"]]
    # The desktop (Xorg, browser, Steam client...) holds a few small slices; only list host
    # programs big enough to matter, such as a game. Unknown usage counts as big.
    big = [p for p in procs if p.container or p.vram_used_gb is None or p.vram_used_gb >= GPU_BIG_GB]
    desktop_gb = round(sum(p.vram_used_gb for p in procs if p not in big), 2)
    procs = big
    others = sorted({p.name for p in procs if not p.container})
    parts = [f"{g['vram_used_gb']} of {g['vram_total_gb']} GB VRAM in use, {g['gpu_utilization_percent']}% busy"]
    parts.append(f"loaded: {'; '.join(loaded)}" if loaded else "no music or image model loaded")
    if others:
        parts.append(f"other GPU programs (not stoppable here): {', '.join(others)}")
    if desktop_gb:
        parts.append(f"desktop apps use {desktop_gb} GB")
    return GpuInfo(
        vram_used_gb=g["vram_used_gb"], vram_total_gb=g["vram_total_gb"],
        gpu_utilization_percent=g["gpu_utilization_percent"], temperature_c=g["temperature_c"],
        models_loaded=loaded, processes=procs, summary="; ".join(parts) + ".")


# Several containers stop in sequence; keep the whole call under Grok's 60s tool limit.
STOP_GRACE_SECONDS = 10


class StopResult(BaseModel):
    stopped: list[str]
    summary: str


@mcp.tool()
async def stop_model(force: bool = False) -> StopResult:
    """Unload every music model from the GPU: stops whichever ACE-Step container is running
    (2B, XL turbo, XL sft, studio mode) and the studio UI, and turns studio mode off. The next
    generate call cold-starts its model again (1-3 minutes). Refuses while a music job is
    running unless force=True, which cuts that job off. Only touches music models; imagegen's
    stop_model unloads the image model."""
    if generation_lock.locked() and not force:
        raise ValueError("A music job is running right now. Wait for it (job_status), "
                         "or call stop_model with force=true to cut it off.")
    names = [*MUSIC_GPU_CONTAINERS, STUDIO_UI_CONTAINER]
    running = [n for n in names if _container_status(n) == "running"]

    async def _stop_all() -> None:
        for name in running:
            docker_client.containers.get(name).stop(timeout=STOP_GRACE_SECONDS)
        STUDIO_XL_FLAG.unlink(missing_ok=True)

    if force:
        await _stop_all()  # the job holding the lock fails when its container goes away
    else:
        async with generation_lock:  # nothing can start between the check and the stop
            await _stop_all()
    if not running:
        return StopResult(stopped=[], summary="No music model was loaded; nothing to stop.")
    return StopResult(stopped=running, summary=f"Stopped {', '.join(running)}; the music model's VRAM is free.")


# --- Driving the Studio UI's own backend (so results land in its library) ----
# The UI's login is a passwordless auto-login (browser auth is the reverse proxy
# at the edge), so music-api asks its backend for a token over the Docker network.
_studio_token: Optional[str] = None


async def _studio_api(method: str, path: str, **kw):
    global _studio_token
    if not (_studio_active() or _xl_active()) or _container_status(STUDIO_UI_CONTAINER) != "running":
        raise ValueError("Studio mode is off. Call studio_on first.")
    async with httpx.AsyncClient(base_url=STUDIO_UI_URL, timeout=30.0) as c:
        for attempt in range(2):
            if _studio_token is None:
                r = await c.post("/api/auth/setup", json={"username": STUDIO_UI_USERNAME})
                r.raise_for_status()
                _studio_token = r.json()["token"]
            r = await c.request(method, path, headers={"Authorization": f"Bearer {_studio_token}"}, **kw)
            if r.status_code == 401 and attempt == 0:
                _studio_token = None
                continue
            r.raise_for_status()
            return r.json()


class StudioJob(BaseModel):
    job_id: str
    status: str = Field(..., description="queued | running | succeeded | failed")
    audio_urls: list[str] = Field(default_factory=list, description=f"Paths on {STUDIO_PUBLIC_URL}; files are also stored by the studio UI")
    error: Optional[str] = None


def _job_from(job_id: str, d: dict) -> StudioJob:
    urls = [STUDIO_PUBLIC_URL + u for u in (d.get("result") or {}).get("audioUrls", [])]
    return StudioJob(job_id=job_id, status=d.get("status", "unknown"), audio_urls=urls, error=d.get("error"))


def _studio_text_fields(style: str, lyrics: str, instrumental: bool) -> dict:
    """Build the style/lyrics/instrumental fields for a studio-ui /api/generate body.

    The studio-ui backend (services/acestep.ts) drops the lyrics entirely when
    instrumental=true, so an instrumental with a word-free [section - cue] structure would
    lose its structure silently. When both are given, send the cues with instrumental=false
    and make sure the caption says "instrumental" instead."""
    has_cues = bool(lyrics.strip()) and lyrics.strip().lower() != "[instrumental]"
    if instrumental and has_cues:
        if "instrumental" not in style.lower():
            style = f"{style.rstrip().rstrip(',')}, instrumental"
        return {"style": style, "lyrics": lyrics, "instrumental": False}
    return {"style": style, "lyrics": lyrics or ("[Instrumental]" if instrumental else ""),
            "instrumental": instrumental}


# Note letter is case-sensitive on purpose so "in a major way" (article "a") is not read as A major.
_KEY_IN_CAPTION = re.compile(r"\b[Ii]n\s+([A-G])([#b♯♭]?)\s+((?i:major|minor))\b")
_KEY_EXPLICIT = re.compile(r"^\s*([A-Ga-g])\s*([#b♯♭]?)\s*(major|minor|maj|min|m)?\s*$", re.IGNORECASE)


def _resolve_key_scale(key_scale: Optional[str], caption: str) -> str:
    """Return the ACE-Step `keyscale` value ("F major", "C# minor", ...) or "" for auto.

    The model is conditioned on a dedicated `keyscale` field ("N/A" when empty), which pins the
    key far better than an `in F major` caption tag alone (2026-10-10: tag-only XL takes read
    on-key 3/11, with the field 6/7). So an explicit key_scale is normalized to ACE-Step's
    format, and when none is given it is taken from an `in <Note> major|minor` tag in the caption."""
    if key_scale and key_scale.strip():
        m = _KEY_EXPLICIT.match(key_scale)
        if m:
            note, acc, mode = m.group(1).upper(), m.group(2), (m.group(3) or "major").lower()
            mode = "minor" if mode in ("minor", "min", "m") else "major"
            return f"{note}{acc} {mode}"
        return key_scale.strip()
    m = _KEY_IN_CAPTION.search(caption or "")
    if m:
        return f"{m.group(1).upper()}{m.group(2)} {m.group(3).lower()}"
    return ""


@mcp.tool()
async def studio_generate(
    style: str,
    lyrics: str = "",
    duration: int = 90,
    instrumental: bool = False,
    bpm: Optional[int] = None,
    key_scale: Optional[str] = None,
    vocal_language: Optional[str] = None,
    seed: Optional[int] = None,
    batch_size: int = 1,
    guidance_scale: float = 7.0,
    infer_method: str = "ode",
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    title: str = "",
    wait: bool = True,
) -> StudioJob:
    """Generate through the Studio UI backend (studio mode must be on). Results are saved
    to the UI's library (visible at the studio web UI) rather than music-api's output dir.
    style = the caption/tags, lyrics = [section]-tagged lyrics. cfg_interval_start/end work
    the same as on generate_song. wait=False returns the job id immediately; poll studio_job."""
    body = {
        "customMode": True, **_studio_text_fields(style, lyrics, instrumental),
        "duration": duration, "batchSize": batch_size,
        "guidanceScale": guidance_scale, "inferMethod": infer_method, "title": title,
        "cfgIntervalStart": cfg_interval_start, "cfgIntervalEnd": cfg_interval_end,
        "thinking": False, "inferenceSteps": 8, "audioFormat": "mp3",
        "randomSeed": seed is None,
    }
    if seed is not None:
        body["seed"] = seed
    if bpm is not None:
        body["bpm"] = bpm
    resolved_key = _resolve_key_scale(key_scale, style)
    if resolved_key:
        body["keyScale"] = resolved_key
    if vocal_language:
        body["vocalLanguage"] = vocal_language
    _set_dcw(True)  # studio runs 2B/XL-turbo, which were tuned with DCW on
    job_id = (await _studio_api("POST", "/api/generate", json=body))["jobId"]
    if not wait:
        return StudioJob(job_id=job_id, status="queued")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        await asyncio.sleep(3)
        d = await _studio_api("GET", f"/api/generate/status/{job_id}")
        if d.get("status") in ("succeeded", "failed"):
            return _job_from(job_id, d)
    return StudioJob(job_id=job_id, status="running", error="still running after 10 min; poll studio_job")


@mcp.tool()
async def studio_job(job_id: str) -> StudioJob:
    """Status/result of a studio_generate job."""
    return _job_from(job_id, await _studio_api("GET", f"/api/generate/status/{job_id}"))


async def _ensure_xl_running(model: str = "turbo") -> None:
    """Start the wanted XL container (turbo acestep-xl, or non-distilled acestep-xl-sft) +
    studio-ui if needed, stopping the other XL variant first. Caller holds generation_lock."""
    want = XL_SFT_CONTAINER if model == "sft" else XL_CONTAINER
    other = XL_CONTAINER if model == "sft" else XL_SFT_CONTAINER
    for name in (want, STUDIO_UI_CONTAINER):
        if _container_status(name) == "missing":
            raise ValueError(
                f"Container {name} does not exist. Create the {name} container "
                "(and studio-ui) before calling this tool."
            )
    if _container_status(other) == "running":
        # Switching variants: both share the "acestep-gradio" alias and the GPU, so never together.
        if _studio_xl_on():
            raise ValueError("studio mode holds the XL container; call studio_off before switching XL variants.")
        docker_client.containers.get(other).stop(timeout=30)
    if _container_status(want) != "running":
        if _container_status(COMFYUI_CONTAINER) == "running":
            raise ValueError("comfyui is running (image generation holds the GPU). Wait for it to idle out, then retry.")
        rest = docker_client.containers.get(ACESTEP_CONTAINER)
        if rest.status == "running":
            rest.stop(timeout=30)
        _refuse_if_studio()  # no await between this check and start()
        docker_client.containers.get(want).start()
    if not await _wait_http(f"{GRADIO_URL}/gradio_api/info", XL_READY_TIMEOUT):
        await _stop_xl()
        raise TimeoutError("acestep-xl did not become ready in time; stopped it (check stack_logs acestep-xl)")
    if _container_status(STUDIO_UI_CONTAINER) != "running":
        docker_client.containers.get(STUDIO_UI_CONTAINER).start()
    if not await _wait_http(f"{STUDIO_UI_URL}/api/generate/health", 90):
        await _stop_xl()
        raise TimeoutError("studio-ui did not become ready in time; stopped XL (check stack_logs studio-ui)")


@mcp.tool()
@background_job
async def generate_song_xl(
    prompt: str,
    lyrics: str = "",
    duration: int = 90,
    seed: Optional[int] = None,
    bpm: Optional[int] = None,
    vocal_language: Optional[str] = None,
    key_scale: Optional[str] = None,
    infer_method: str = "ode",
    instrumental: bool = False,
    title: str = "",
    model: str = "sft",
    guidance_scale: Optional[float] = None,
    inference_steps: Optional[int] = None,
    shift: Optional[float] = None,
    use_adg: Optional[bool] = None,
    thinking: Optional[bool] = None,
    lm_codes_strength: Optional[float] = None,
    lm_temperature: Optional[float] = None,
    lm_cfg_scale: Optional[float] = None,
    lm_negative_prompt: Optional[str] = None,
) -> MusicResult:
    """thinking=True turns on ACE-Step's 5Hz LM planner (needs the container started with an LM; slower). Omitted = env XL_THINKING_DEFAULT (0/1).
    model="sft" (DEFAULT since 2026-10-11) is the non-distilled 4B XL: 50 steps, ~50s per 175s
    track, wider dynamics and 99%-in-key vocals; DCW is switched off automatically for it. Keep
    duration <= 185s on sft (VRAM headroom is thin). model="turbo" is the CFG-distilled 8-step XL
    (~15s per track, needed for duration > 185s). guidance_scale (typical 3-9, default 7),
    inference_steps (default 50 sft / 8 turbo) and shift (default 3) only matter for sft; turbo
    ignores guidance. Switching model stops the other XL variant (container swap, ~2-3 min).
    Generate a song with the 4B XL turbo model (unquantized bf16, DiT swapped through host
    RAM), which is much steadier on rhythm and vocal phrasing than generate_song's 2B. Same
    prompt/lyrics conventions as generate_song; the result lands in the same place
    (/music-api/audio/ URL, 30-day retention). key_scale ("F major", "D minor") is sent as
    ACE-Step's dedicated keyscale field; if omitted it is taken from an "in <key> major|minor"
    tag in the prompt, so naming the key in the prompt is enough. Starts acestep-xl + the studio UI on demand (cold start ~2-3 min) and
    stops them after XL_IDLE_TIMEOUT idle (default 15 min); a later generate_song call
    switches back to 2B automatically. Refused while studio mode is on. No
    guidance_scale/cfg params: turbo models ignore them. instrumental=True with lyrics
    keeps the lyrics as a word-free [section - cue] structure (sent with the flag off and
    "instrumental" in the caption, since the backend would otherwise discard them)."""
    global xl_last_used
    duration = max(30, min(duration, 240))
    _refuse_if_studio()
    async with generation_lock:
        _refuse_if_studio()
        xl_last_used = time.monotonic()
        model = "turbo" if str(model).lower() == "turbo" else "sft"
        if model == "sft" and duration > 185:
            raise ValueError("sft peaks ~11.8GB of the 11.6GiB card and is only verified to 175s; use duration <= 185 "
                             "or model=\"turbo\" for longer tracks")
        await _ensure_xl_running(model)
        _set_dcw(model != "sft")
        steps = inference_steps if inference_steps else (50 if model == "sft" else 8)
        body = {
            "customMode": True, **_studio_text_fields(prompt, lyrics, instrumental),
            "duration": duration, "batchSize": 1,
            "inferMethod": infer_method, "title": title,
            "thinking": XL_THINKING_DEFAULT if thinking is None else bool(thinking), "inferenceSteps": steps, "audioFormat": "mp3",
            "randomSeed": seed is None,
            "useCotCaption": XL_COT_CAPTION,
        }
        if model == "sft":
            body["guidanceScale"] = guidance_scale if guidance_scale is not None else 7.0
        # The studio-ui backend defaults shift to 3.0 (turbo's distillation value); the ACE-Step docs list 1.0 for
        # the non-distilled models, so it is exposed for the sft experiments (2026-10-10).
        if shift is not None:
            body["shift"] = shift
        elif model == "sft":
            body["shift"] = 3.0  # the value the DCW-off sft tests used (ACE-Step's own sft UI default)
        if use_adg is not None:
            body["useAdg"] = use_adg
        # LM experiment knobs (2026-10-18); only matter when thinking is on. lm_codes_strength is ACE-Step's
        # audio_cover_strength: how strongly the LM's audio codes steer the DiT (1.0 = fully).
        if lm_codes_strength is not None:
            body["audioCoverStrength"] = lm_codes_strength
        if lm_temperature is not None:
            body["lmTemperature"] = lm_temperature
        if lm_cfg_scale is not None:
            body["lmCfgScale"] = lm_cfg_scale
        if lm_negative_prompt:
            body["lmNegativePrompt"] = lm_negative_prompt
        if seed is not None:
            body["seed"] = seed
        if bpm is not None:
            body["bpm"] = bpm
        resolved_key = _resolve_key_scale(key_scale, prompt)
        if resolved_key:
            body["keyScale"] = resolved_key
        if vocal_language:
            body["vocalLanguage"] = vocal_language
        job_id = (await _studio_api("POST", "/api/generate", json=body))["jobId"]
        d: dict = {}
        deadline = time.monotonic() + (2400 if body["thinking"] else 600)
        while time.monotonic() < deadline:
            await asyncio.sleep(3)
            d = await _studio_api("GET", f"/api/generate/status/{job_id}")
            if d.get("status") in ("succeeded", "failed"):
                break
        xl_last_used = time.monotonic()
        if d.get("status") != "succeeded":
            raise ValueError(f"XL generation did not succeed (status={d.get('status')}, error={d.get('error')})")
        paths = (d.get("result") or {}).get("audioUrls") or []
        if not paths:
            raise ValueError("XL generation succeeded but returned no audio path")
        async with httpx.AsyncClient(base_url=STUDIO_UI_URL, timeout=60.0) as c:
            r = await c.get(paths[0], headers={"Authorization": f"Bearer {_studio_token}"})
            r.raise_for_status()
            audio_bytes = r.content

    out_name = _planned_out_name.get() or f"{uuid.uuid4().hex}.mp3"
    (OUTPUT_DIR / out_name).write_bytes(audio_bytes)
    return MusicResult(audio_url=AUDIO_URL_BASE + out_name)


@mcp.custom_route("/studio/status", methods=["GET"])
async def studio_status_route(request: Request) -> JSONResponse:
    return JSONResponse(_studio_status().model_dump())


@mcp.custom_route("/studio/on", methods=["POST"])
async def studio_on_route(request: Request) -> JSONResponse:
    try:
        model = request.query_params.get("model", "2b")
        return JSONResponse((await _studio_on(True, model)).model_dump())
    except (ValueError, TimeoutError) as e:
        return JSONResponse({"error": str(e)}, status_code=409)


@mcp.custom_route("/studio/off", methods=["POST"])
async def studio_off_route(request: Request) -> JSONResponse:
    return JSONResponse((await _studio_off()).model_dump())


# Public via the reverse proxy, so serve only names this server generates (uuid4 hex + .mp3), never
# other files that land in OUTPUT_DIR such as the .studio_xl flag (audit 2026-10-16).
AUDIO_NAME_RE = re.compile(r"[0-9a-f]{32}\.mp3")


@mcp.custom_route("/audio/{filename}", methods=["GET"])
async def get_audio(request: Request):
    filename = request.path_params["filename"]
    if not AUDIO_NAME_RE.fullmatch(filename):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.exists():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(path, media_type="audio/mpeg")


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
