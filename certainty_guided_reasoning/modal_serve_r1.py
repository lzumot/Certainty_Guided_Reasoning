"""Deploy target: `modal deploy certainty_guided_reasoning/modal_serve_r1.py`.

DeepSeek-R1-Distill-Qwen-1.5B on SGLang/Modal for the GSM8K run. H100: a 1.5B
decode is bandwidth-bound, so an L4 would be the bottleneck. Cold start, so
nothing bills while idle. Stop billing: `modal app stop cgr-sglang-r1`.
"""

import modal

N_MINUTES = 60
CACHE_MOUNT = "/cache"

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GPU = "H100"
MIN_CONTAINERS = 0  # cold start: nothing warm while idle
MAX_CONTAINERS = 8  # 8xH100 parallelises the 64-seed capture
CONTEXT_LENGTH = 16384  # the GSM8K protocol needs far less than the 131k default
PORT = 8000

hf_cache = modal.Volume.from_name("cgr-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("sglang[all]", "huggingface_hub[hf_xet]")
    .env({
        "HF_HOME": f"{CACHE_MOUNT}/huggingface",
        # Persist caches so cold starts skip recompilation.
        "SGLANG_CACHE_ROOT": f"{CACHE_MOUNT}/sglang",
        "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_MOUNT}/inductor",
    })
)

app = modal.App("cgr-sglang-r1")


@app.function(
    image=image,
    gpu=GPU,
    volumes={CACHE_MOUNT: hf_cache},
    scaledown_window=30 * N_MINUTES,
    # Bounds the WHOLE container lifetime for a @web_server function; a low
    # value silently caps startup_timeout too.
    timeout=24 * 60 * N_MINUTES,
    secrets=[modal.Secret.from_name("hf-token")],
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
)
# Matches the client-side worker count so requests are never serialised behind
# a lower server-side cap.
@modal.concurrent(max_inputs=256, target_inputs=160)
@modal.web_server(port=PORT, startup_timeout=30 * N_MINUTES)
def serve():
    import subprocess

    cmd = (
        f"python -m sglang.launch_server --model-path {MODEL_ID} "
        f"--host 0.0.0.0 --port {PORT} "
        f"--context-length {CONTEXT_LENGTH} "
        "--enable-custom-logit-processor "
        "--reasoning-parser deepseek-r1 "
        "--speculative-num-steps 0"
    )
    subprocess.Popen(cmd, shell=True)
