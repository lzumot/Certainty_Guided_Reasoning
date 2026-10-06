"""Deploy an SGLang OpenAI-compatible endpoint on Modal.

Usage:
    uv run modal deploy certainty_guided_reasoning/modal_serve.py
    # 35B-A3B (H200):
    GPU=H200 MIN_CONTAINERS=4 MAX_CONTAINERS=4 MODEL_ID=Qwen/Qwen3.5-35B-A3B \
        uv run modal deploy certainty_guided_reasoning/modal_serve.py

SGLang is used (not vLLM) for its support of the Qwen3.5 hybrid architecture.
"""

import os

import modal

N_MINUTES = 60
CACHE_MOUNT = "/cache"
PORT = 8000

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
GPU = os.environ.get("GPU", "B200")
MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", "8"))
MAX_CONTAINERS = os.environ.get("MAX_CONTAINERS")  # optional cap; e.g. 4 for 35B
APP_NAME = os.environ.get("APP_NAME", "cgr-sglang")

hf_cache = modal.Volume.from_name("cgr-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("sglang[all]", "huggingface_hub[hf_xet]")
    .env({
        "HF_HOME": f"{CACHE_MOUNT}/huggingface",
        # Persist caches across cold starts.
        "SGLANG_CACHE_ROOT": f"{CACHE_MOUNT}/sglang",
        "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_MOUNT}/inductor",
    })
)

app = modal.App(APP_NAME)

_function_kwargs = {
    "image": image,
    "gpu": GPU,
    "volumes": {CACHE_MOUNT: hf_cache},
    "scaledown_window": 30 * N_MINUTES,
    "timeout": 24 * 60 * N_MINUTES,  # whole-container lifetime
    "secrets": [modal.Secret.from_name("hf-token")],
    "min_containers": MIN_CONTAINERS,
}
if MAX_CONTAINERS:
    _function_kwargs["max_containers"] = int(MAX_CONTAINERS)


@app.function(**_function_kwargs)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=30 * N_MINUTES)
def serve():
    import subprocess

    cmd = (
        f"python -m sglang.launch_server --model-path {MODEL_ID} "
        f"--host 0.0.0.0 --port {PORT} "
        "--enable-custom-logit-processor "
        "--reasoning-parser qwen3 "
        # Qwen3.5 MTP speculative path does not compose with stateful logits
        # processors; disabled until verified.
        "--speculative-num-steps 0"
    )
    subprocess.Popen(cmd, shell=True)
