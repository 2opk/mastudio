## Overview
Director–Generator MAS for image prompt iteration. Uses LiteLLM for chat models and an SDXL stub for image generation. Defaults to a local vLLM server running Qwen2.5-VL-7B-Instruct, but a ChatGPT preset is available via `config/system_config_openai.yaml`.

## Prereqs
- Python 3.13 (conda env recommended)
- `pip install uv; uv pip install -r requirements.txt`
- Local OpenAI-compatible endpoint (vLLM) at `http://127.0.0.1:8000/v1`

## Quickstart
1) Export env (dummy key is fine for local vLLM), or set `.env` by copying and modifying given `.env.example`:
   ```bash
   export OPENAI_API_KEY=EMPTY
   export VLLM_API_BASE=http://127.0.0.1:8000/v1
   ```
2) Run vLLM:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model qwen/Qwen2.5-VL-7B-Instruct \
     --host 0.0.0.0 --port 8000
   ```
   Wait for the server to print the model ID (`qwen/Qwen2.5-VL-7B-Instruct`).
3) Execute a run:
   ```bash
   python -m src.main "sample prompt"
   ```
   - Use a different config with `--config config/system_config_openai.yaml` to hit the ChatGPT API (requires `OPENAI_API_KEY`).
   - Progress logs print each phase and LLM call (system/user previews and replies).
   - Output JSON and mock images land in `output/`.

## Configuration
- `config/system_config.yaml` controls model aliases, vLLM endpoint, SDXL mode, loop limits, and recursion limit (defaults: `max_loops=10`, `recursion_limit=1000`, SDXL `mode=mock`).
- `config/directors.json` and `config/generators.json` define agent personas; prompts live under `prompts/`.
- For SDXL: `sdxl.mode` can be `mock` (placeholder PNGs), `api` (Stability API; requires `STABILITY_API_KEY`), or `local` (loads the SDXL base pipeline to `sdxl.device`).
- SDXL failover: set `sdxl.failover_mode: mock` (or env `SDXL_FAILOVER_MODE=mock`) to fall back to mock images if the SDXL server returns an error.
- CLI accepts `--config` to point at an alternate system config and `--max-iteration/--max-iterations` to override loop count.

## ChatGPT API preset
- Set `OPENAI_API_KEY` in your environment (no `VLLM_API_BASE` needed).
- Use `config/system_config_openai.yaml` to map roles to `gpt-4o` / `gpt-4o-mini`.
- Example with a prompt file:
  ```bash
  python -m src.main -f audio2txt/processed/0000_45-55.json \
    --max-iterations 5 \
    --output-dir output_chatgpt/test_0000_45-55 \
    --config config/system_config_openai.yaml
  ```

## SDXL HTTP server (reuse one pipeline, fan out clients)
- Start one server per GPU; the pipeline loads once and stays hot:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python -m scripts.sdxl_server --port 7000 --device cuda:0
  CUDA_VISIBLE_DEVICES=1 python -m scripts.sdxl_server --port 7001 --device cuda:0
  ```
- Point clients with `SDXL_API_BASE=http://127.0.0.1:7000` (env overrides config and routes SDXL calls to the server). Set `sdxl.mode: api` in your config if you prefer an explicit toggle.
- To keep jobs alive even if the server hiccups, add `sdxl.failover_mode: mock` in the config or export `SDXL_FAILOVER_MODE=mock` so failed requests fall back to mock images instead of crashing the run.
- Parallel example that shards across two SDXL servers:
  ```bash
  export GPUS=\"0 1\"
  find audio2txt/processed -maxdepth 1 -name '*.json' -print0 |
    parallel -0 --jobs 4 --lb --env GPUS '
      file=\"{}\"
      gpus=($GPUS); n=${#gpus[@]}
      slot={#}; idx=$(( (slot-1) % n ))
      sdxl_port=$((7000 + idx))
      base=$(basename \"$file\" .json)
      out_dir=\"output_chatgpt/output_$(date +%Y%m%d_%H%M%S_%N)_${base}\"
      mkdir -p \"$out_dir\"
      SDXL_API_BASE=http://127.0.0.1:${sdxl_port} \
      python -m src.main -f \"$file\" --max-iterations 5 --output-dir \"$out_dir\" \
        --config config/system_config_openai.yaml \
        > \"$out_dir/run.log\" 2>&1
    '
  ```
- Health check: `curl http://127.0.0.1:7000/docs` to confirm the server is up.

## Notes
- Local vLLM must advertise a model id matching the alias in `config/system_config.yaml` (case-sensitive).
- Recursion errors: increase `recursion_limit` in `config/system_config.yaml` or adjust `evaluation_threshold`/`max_loops`.
