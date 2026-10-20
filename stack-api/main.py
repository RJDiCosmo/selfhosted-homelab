"""Stack Status MCP server — read-only visibility into the Docker Compose stack for LLM agents.

Talks to Docker only through docker-socket-proxy-ro, which is configured with
CONTAINERS=1 and no POST — i.e. list/inspect/logs only. There is no code path
in this service that can start, stop, restart, or exec into a container.
"""

import os
import re

import docker
import psutil
import pynvml
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

docker_client = docker.DockerClient(base_url=os.environ["DOCKER_HOST"])

# nvmlInit() opens a handle to the driver once at import time -- cheap, and
# every gpu_status() call just reuses it rather than re-init/shutdown per call.
pynvml.nvmlInit()

# Container names are Docker identifiers (letters, digits, underscore, dot, dash).
CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

mcp = FastMCP(
    name="Stack Status",
    instructions=(
        "Read-only status and logs for the containers in this self-hosted "
        "Docker Compose stack. Cannot start, stop, restart, or modify anything."
    ),
    host="0.0.0.0",
    port=8020,
)


class ContainerStatus(BaseModel):
    name: str
    service: str
    image: str
    status: str
    health: str | None = None
    restart_count: int
    started_at: str


# FastMCP serializes a bare list[Model] return as one MCP content block *per item* —
# fine for a REST JSON body, but overwhelms a small model's ability to track many
# separate tool-result blocks in one turn (confirmed: a 15-container stack_status
# produced 15 blocks, contributing to nemotron dropping its final answer). Wrapping
# the list in a single object keeps each tool call to one content block.
class ContainerStatusList(BaseModel):
    containers: list[ContainerStatus]


class LogLines(BaseModel):
    container: str
    lines: list[str]


class SystemResources(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float


class GpuStatus(BaseModel):
    name: str
    gpu_utilization_percent: int
    vram_used_gb: float
    vram_total_gb: float
    vram_percent: float
    temperature_c: int


class GpuProcess(BaseModel):
    pid: int
    name: str
    vram_used_gb: float


class GpuProcessList(BaseModel):
    processes: list[GpuProcess]


@mcp.tool()
async def stack_status() -> ContainerStatusList:
    """List all stack containers and their current status, health, and restart count."""
    result = []
    for container in docker_client.containers.list(all=True):
        attrs = container.attrs
        labels = attrs.get("Config", {}).get("Labels", {}) or {}
        health = attrs.get("State", {}).get("Health", {}).get("Status")
        result.append(
            ContainerStatus(
                name=container.name,
                service=labels.get("com.docker.compose.service", container.name),
                image=attrs.get("Config", {}).get("Image", "unknown"),
                status=container.status,
                health=health,
                restart_count=attrs.get("RestartCount", 0),
                started_at=attrs.get("State", {}).get("StartedAt", ""),
            )
        )
    return ContainerStatusList(containers=result)


@mcp.tool()
async def stack_logs(container_name: str, tail: int = 100) -> LogLines:
    """Tail recent log lines for one container. tail is capped at 500."""
    if not CONTAINER_NAME_RE.match(container_name):
        raise ValueError("Invalid container name")
    tail = max(1, min(tail, 500))

    try:
        container = docker_client.containers.get(container_name)
    except docker.errors.NotFound:
        raise ValueError("Container not found")

    raw = container.logs(tail=tail, timestamps=True).decode("utf-8", errors="replace")
    return LogLines(container=container_name, lines=raw.splitlines())


@mcp.tool()
async def system_resources() -> SystemResources:
    """Host CPU and RAM usage (not just this container's cgroup)."""
    # interval=0.5 blocks briefly to sample over a window rather than returning
    # a meaningless 0.0 from a single instantaneous read.
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    return SystemResources(
        cpu_percent=cpu,
        ram_used_gb=round(mem.used / 1024**3, 2),
        ram_total_gb=round(mem.total / 1024**3, 2),
        ram_percent=mem.percent,
    )


@mcp.tool()
async def gpu_status() -> GpuStatus:
    """GPU utilization and VRAM usage for the host's NVIDIA card."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    return GpuStatus(
        name=name,
        gpu_utilization_percent=util.gpu,
        vram_used_gb=round(mem.used / 1024**3, 2),
        vram_total_gb=round(mem.total / 1024**3, 2),
        vram_percent=round(mem.used / mem.total * 100, 1),
        temperature_c=temp,
    )


@mcp.tool()
async def gpu_processes() -> GpuProcessList:
    """List processes currently using the GPU, with their VRAM usage."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    result = []
    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        try:
            with open(f"/proc/{proc.pid}/comm") as f:
                name = f.read().strip()
        except FileNotFoundError:
            # Process exited between the NVML query and this read.
            name = "unknown"
        used = proc.usedGpuMemory or 0
        result.append(GpuProcess(pid=proc.pid, name=name, vram_used_gb=round(used / 1024**3, 2)))
    return GpuProcessList(processes=result)


# Docker's cgroup path for a containerized process ends in the 64-hex container id
# (cgroup v2: "0::/system.slice/docker-<id>.scope").
DOCKER_ID_RE = re.compile(r"[0-9a-f]{64}")


def _container_of(pid: int) -> str | None:
    """Name of the container a host pid runs in, or None for a host process (e.g. a game)."""
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            m = DOCKER_ID_RE.search(f.read())
    except OSError:
        return None
    if not m:
        return None
    try:
        return docker_client.containers.get(m.group(0)).name
    except docker.errors.DockerException:
        return m.group(0)[:12]


# Not an MCP tool: a plain-JSON GPU snapshot for music-api/image-api's gpu_status tools
# (added 2026-10-19), so those internet-reachable services get GPU visibility without a
# GPU reservation of their own. Read-only like everything else here; keep it on the
# internal Docker network and localhost only (do not publish it through the reverse proxy).
@mcp.custom_route("/gpu", methods=["GET"])
async def gpu_route(request: Request) -> JSONResponse:
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    procs, seen = [], set()
    # Compute = CUDA jobs (ComfyUI, ACE-Step); graphics = games and the desktop.
    for proc in (pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                 + pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)):
        if proc.pid in seen:
            continue
        seen.add(proc.pid)
        try:
            with open(f"/proc/{proc.pid}/comm") as f:
                name = f.read().strip()
        except FileNotFoundError:
            name = "unknown"
        used = proc.usedGpuMemory
        procs.append({"pid": proc.pid, "name": name, "container": _container_of(proc.pid),
                      "vram_used_gb": round(used / 1024**3, 2) if used else None})
    return JSONResponse({**(await gpu_status()).model_dump(), "processes": procs})


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
