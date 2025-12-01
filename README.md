## Overview
Director–Generator MAS for image prompt iteration. Uses LiteLLM for chat models and an SDXL stub for image generation. Currently wired to a local vLLM server running Qwen2.5-VL-7B-Instruct.

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
   - Progress logs print each phase and LLM call (system/user previews and replies).
   - Output JSON and mock images land in `output/`.

## Configuration
- `config/system_config.yaml` controls model aliases, vLLM endpoint, SDXL mode, loop limits, and recursion limit (defaults: `max_loops=10`, `recursion_limit=1000`, SDXL `mode=mock`).
- `config/directors.json` and `config/generators.json` define agent personas; prompts live under `prompts/`.
- For real SDXL calls, set `sdxl.mode: stability` and `STABILITY_API_KEY` in your environment.

## Notes
- Local vLLM must advertise a model id matching the alias in `config/system_config.yaml` (case-sensitive).
- Recursion errors: increase `recursion_limit` in `config/system_config.yaml` or adjust `evaluation_threshold`/`max_loops`.