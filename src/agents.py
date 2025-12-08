import base64
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

import litellm
from rich.console import Console
from rich.text import Text

from .utils import (
    DirectorMeta,
    GeneratorAgentMeta,
    SystemConfig,
    load_agent_aliases,
    load_directors_auto,
    load_generator_pool,
    load_prompt,
)


class MASState(TypedDict, total=False):
    iteration: int
    user_prompt: str
    mood_report: Dict[str, Any]
    visual_blueprint: Dict[str, Any]
    conceptual_elements: Dict[str, Any]
    final_draft: Dict[str, Any]
    squad_selection_instructions: Dict[str, Any]
    squad_assignments: Dict[str, List[Dict[str, Any]]]
    squad_prompt_map: Dict[str, str]
    generator_prompts: List[str]
    images: List[str]
    feedback: str
    scores: List[float]
    history: List[Dict[str, str]]
    critic_results: List[Dict[str, Any]]
    creative_index_avg: float
    squad_ci_avg: Dict[str, float]
    squad_status: Dict[str, str]
    critic_feedback: str
    t0: float


class ModelRouter:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.console = Console()
        self._logged_models = set()

    def _preview(self, text: Any, limit: Optional[int] = None) -> str:
        if text is None:
            return ""
        if isinstance(text, list):
            texts: List[str] = []
            for part in text:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = str(part.get("text", ""))
                    texts.append(t)
            combined = "\n\n".join(texts) if texts else f"[multimodal {len(text)} parts]"
            return self._sanitize_preview(combined, limit)
        return self._sanitize_preview(str(text), limit)

    def _sanitize_preview(self, raw: str, limit: Optional[int]) -> str:
        s = raw.replace("\\n", "\n").replace('\\"', '"')
        if limit is None:
            return s
        return s if len(s) <= limit else s[: max(limit - 3, 0)] + "..."

    def _api_base_for_model(self, model: str) -> Optional[str]:
        if model.startswith("gemini/"):
            return os.getenv("GEMINI_API_BASE") or None
        if model == self.config.open_source_model:
            return self.config.vllm.api_base
        return None

    def call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        agent_label: Optional[str] = None,
        user_content: Optional[Union[str, List[Dict[str, Any]]]] = None,
    ) -> str:
        resolved_model = self.config.resolve_model(model)
        content = user_content if user_content is not None else user_prompt
        kwargs = {
            "model": resolved_model,
            "api_base": self._api_base_for_model(resolved_model),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
        }
        # Gemini models can require an explicit API version; allow overriding via env.
        if resolved_model.startswith("gemini/"):
            kwargs["api_version"] = os.getenv("GEMINI_API_VERSION", "v1")
        # Local OpenAI-compatible endpoints (vLLM) usually need a dummy API key.
        if kwargs["api_base"] and not kwargs.get("api_key"):
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY") or "EMPTY"
        # If talking to an OpenAI-compatible endpoint, force provider for litellm.
        if kwargs["api_base"] and not resolved_model.startswith(("gemini/", "anthropic/", "openai/")):
            kwargs["custom_llm_provider"] = "openai"

        self._log_model_once(resolved_model, kwargs["api_base"])
        response = litellm.completion(**kwargs)
        content = response.choices[0].message["content"]
        if agent_label:
            self._print_chat_line(agent_label, "", self._preview(content))
        return content

    def _log_model_once(self, model: str, api_base: Optional[str]) -> None:
        key = (model, api_base)
        if key in self._logged_models:
            return
        self._logged_models.add(key)
        line = Text(f"[model] {model}", style="dim")
        if api_base:
            line.append(f" via {api_base}", style="dim")
        self.console.print(line)

    def _print_chat_line(self, agent_label: str, direction: str, text: str) -> None:
        lower_label = agent_label.lower()
        is_generator = "squad" in lower_label or "generator" in lower_label
        header = Text(f"[{agent_label}] {direction}", style="bold magenta" if is_generator else "bold cyan")
        body = Text(f"{text}", style="grey70")
        self.console.print(header)
        self.console.print(body)
        self.console.print()  # blank line for readability


def _safe_json_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip("` \n")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        return {"raw": raw.strip()}


def _clean_prompt_text(raw: str) -> str:
    text = raw.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
        text = text.replace("```", "").strip()
    # If looks like JSON with 'prompt', attempt to extract.
    if "{" in text and "prompt" in text:
        parsed = _safe_json_parse(text)
        if isinstance(parsed, dict) and parsed.get("prompt"):
            text = str(parsed.get("prompt", "")).strip()
    # Remove surrounding quotes.
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def _finalize_prompt_text(raw: str, max_len: int = 77) -> str:
    text = _clean_prompt_text(raw)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if len(tokens) > max_len:
        text = " ".join(tokens[:max_len])
    return text


def creation_director_phase(state: MASState, config: SystemConfig, router: ModelRouter) -> MASState:
    router.console.print(
        Text(
            f"[phase:director_creation] base_prompt={router._preview(state.get('user_prompt',''))}",
            style="bold white",
        )
    )
    aliases = load_agent_aliases()
    directors = load_directors_auto(aliases)
    if len(directors) < 4:
        raise ValueError("Director config must include four agents for creation/critic modes.")
    common_prompt = load_prompt("prompts/directors/creation_mode/COMMON_INSTRUCTION_CREATION_D.md")
    history_entries: List[Dict[str, str]] = state.get("history", []).copy()

    base_prompt = state.get("user_prompt", "")
    prior_feedback = state.get("critic_feedback", "none")
    pool = load_generator_pool()
    agent_catalog: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for meta in pool:
        agent_catalog[meta.category or "uncategorized"].append(
            {
                "id": meta.id,
                "name": meta.name,
                "display_name": meta.display_name,
                "keywords": meta.keywords,
                "category": meta.category,
            }
        )

    def _call_director(meta: DirectorMeta, payload: str) -> str:
        persona = load_prompt(meta.creation_prompt)
        system_prompt = f"{common_prompt}\n\n{persona}\nOutput strict JSON only."
        return router.call(
            model=config.director_model,
            system_prompt=system_prompt,
            user_prompt=payload,
            temperature=0.4,
            agent_label=meta.display_name or meta.name,
        )

    # Music Interpreter
    mi_payload = (
        f"Base brief: {base_prompt}\n"
        f"Feedback to consider: {prior_feedback}\n"
        "You are first in the chain. Produce Mood Report JSON."
    )
    mi_output_raw = _call_director(directors[0], mi_payload)
    mi_output = _safe_json_parse(mi_output_raw)
    history_entries.append({"role": directors[0].display_name or directors[0].name, "content": mi_output_raw})

    # Visual Director
    vd_payload = (
        f"Base brief: {base_prompt}\n"
        f"Mood Report JSON: {json.dumps(mi_output, ensure_ascii=False)}\n"
        "Build a Visual Blueprint as JSON. Do not overwrite Mood Report."
    )
    vd_output_raw = _call_director(directors[1], vd_payload)
    vd_output = _safe_json_parse(vd_output_raw)
    history_entries.append({"role": directors[1].display_name or directors[1].name, "content": vd_output_raw})

    # Concept Architect
    ca_payload = (
        f"Base brief: {base_prompt}\n"
        f"Mood Report JSON: {json.dumps(mi_output, ensure_ascii=False)}\n"
        f"Visual Blueprint JSON: {json.dumps(vd_output, ensure_ascii=False)}\n"
        "Populate the scene with conceptual elements and a visual reference."
    )
    ca_output_raw = _call_director(directors[2], ca_payload)
    ca_output = _safe_json_parse(ca_output_raw)
    history_entries.append({"role": directors[2].display_name or directors[2].name, "content": ca_output_raw})

    # Orchestrator
    orch_payload = (
        f"Base brief: {base_prompt}\n"
        f"Mood Report JSON: {json.dumps(mi_output, ensure_ascii=False)}\n"
        f"Visual Blueprint JSON: {json.dumps(vd_output, ensure_ascii=False)}\n"
        f"Conceptual Elements JSON: {json.dumps(ca_output, ensure_ascii=False)}\n"
        f"Agent Catalog JSON (grouped by category): {json.dumps(agent_catalog, ensure_ascii=False)}\n"
        "Synthesize into final_draft AND squad_selection_instructions AND squad_assignments JSON.\n"
        "For squad_selection_instructions, derive target_keywords FRESH from the current Final Draft + Mood/Visual/Concept data (no hardcoded lists). Each list should be 10-20 concise cues:\n"
        "  - squad_a_harmonic: detailed perceptual/visual/music cues that best align with the current draft (mood, instrumentation, lighting, framing, texture, tempo, key, energy).\n"
        "  - squad_b_conflict: CLEAR opposites of the current draft (brightness vs darkness, energetic vs calm, fast vs slow, major vs minor, harsh vs soft, chaotic vs composed, vivid vs muted, aggressive vs gentle). Conflict should intentionally disagree with the prompt mood.\n"
        "For squad_assignments:\n"
        "- Each squad must contain exactly one agent per category present in the catalog.\n"
        "- squad_a_harmonic: pick the closest agent per category to the harmonic target_keywords.\n"
        "- squad_b_conflict: pick the closest agent per category to the conflict target_keywords, avoiding all harmonic picks, and avoid agents whose keywords overlap with harmonic cues (melancholic/soft/diffused/etc.).\n"
        "- squad_c_random: pick randomly per category, excluding any agent already used in harmonic or conflict.\n"
        "- Return only agent ids in each squad list (category order does not matter)."
    )
    orchestrator_raw = _call_director(directors[3], orch_payload)
    orchestrator_json = _safe_json_parse(orchestrator_raw)
    history_entries.append({"role": directors[3].display_name or directors[3].name, "content": orchestrator_raw})

    final_draft = orchestrator_json.get("final_draft", {}) if isinstance(orchestrator_json, dict) else {}
    squad_instructions = orchestrator_json.get("squad_selection_instructions", {}) if isinstance(orchestrator_json, dict) else {}
    provided_assignments = orchestrator_json.get("squad_assignments", {}) if isinstance(orchestrator_json, dict) else {}

    id_map = {m.id: m for m in pool}
    squad_assignments: Dict[str, List[GeneratorAgentMeta]] = {}
    for squad_name in ["harmonic", "conflict", "random"]:
        metas: List[GeneratorAgentMeta] = []
        seen_ids: set = set()
        for aid in provided_assignments.get(squad_name, []) or []:
            meta = id_map.get(aid)
            if meta and meta.id not in seen_ids:
                metas.append(meta)
                seen_ids.add(meta.id)
        squad_assignments[squad_name] = metas

    used_orchestrator = all(squad_assignments.get(k) for k in ["harmonic", "conflict", "random"])
    if not used_orchestrator:
        raise ValueError("Orchestrator must return complete squad_assignments with agent ids per category.")

    # Convert to plain dicts for state storage.
    squad_assignments_dict: Dict[str, List[Dict[str, Any]]] = {}
    for squad, metas in squad_assignments.items():
        squad_assignments_dict[squad] = [
            {
                "id": m.id,
                "name": m.name,
                "prompt": m.prompt,
                "display_name": m.display_name,
                "keywords": m.keywords,
                "category": m.category,
            }
            for m in metas
        ]

    selection_record = {
        "instructions": squad_instructions,
        "orchestrator_assignments": provided_assignments,
        "assignments": squad_assignments_dict,
        "used_orchestrator_assignments": used_orchestrator,
    }
    history_entries.append(
        {
            "role": "Squad Selector",
            "content": json.dumps(selection_record, ensure_ascii=False, indent=2),
        }
    )

    return {
        **state,
        "mood_report": mi_output,
        "visual_blueprint": vd_output,
        "conceptual_elements": ca_output,
        "final_draft": final_draft,
        "squad_selection_instructions": squad_instructions,
        "squad_assignments": squad_assignments_dict,
        "squad_selection": selection_record,
        "squad_status": state.get("squad_status") or {"harmonic": "active", "conflict": "active", "random": "active"},
        "history": history_entries,
        "feedback": orchestrator_raw,
    }


def _agent_from_dict(data: Union[Dict[str, Any], GeneratorAgentMeta]) -> GeneratorAgentMeta:
    if isinstance(data, GeneratorAgentMeta):
        return data
    return GeneratorAgentMeta(**data)


def _aggregate_squad_prompt(
    squad_name: str,
    final_draft: Dict[str, Any],
    contributions: List[Dict[str, str]],
    feedback: str,
    router: ModelRouter,
    config: SystemConfig,
) -> str:
    system_prompt = (
        "You are the Squad Synthesizer. Merge the dialogue into ONE SDXL-ready prompt, <=77  tokens"
        "Preserve Final Draft intent, include camera/lighting/color/style, and bake in feedback."
        "Return ONLY the prompt as plain text (no JSON, no labels, no code fences)."
    )
    convo_text = "\n\n".join([f"{c['agent']}: {c['content']}" for c in contributions])
    user_prompt = (
        f"Squad: {squad_name}\n"
        f"Final Draft: {json.dumps(final_draft, ensure_ascii=False)}\n"
        f"Feedback to apply: {feedback}\n"
        f"Dialogue:\n{convo_text}\n"
        "Return one plain-text prompt <=77 tokens"
    )
    raw = router.call(
        model=config.generator_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.6,
        agent_label=f"{squad_name.title()} Synth",
    )
    return _finalize_prompt_text(raw, max_len=77)


def generator_phase(state: MASState, config: SystemConfig, router: ModelRouter) -> MASState:
    router.console.print(Text(f"[phase:generator] iteration={state.get('iteration',0)}", style="bold white"))
    common_prompt = load_prompt("prompts/generators/creation_mode/COMMON_INSTRUCTION_CREATION_G.md")
    history_entries: List[Dict[str, str]] = state.get("history", []).copy()
    final_draft = state.get("final_draft", {})
    feedback = state.get("critic_feedback", "")
    base_prompt = state.get("user_prompt", "")
    squad_status = state.get("squad_status") or {"harmonic": "active", "conflict": "active", "random": "active"}
    router.console.print(Text(f"squad_status: {squad_status}", style="dim"))

    squad_assignments_raw = state.get("squad_assignments") or {}
    if not squad_assignments_raw:
        raise ValueError("Missing squad_assignments in state; orchestrator must supply them.")

    prompts_out: List[str] = []
    squad_prompt_map: Dict[str, str] = {}
    prompt_squad_sequence: List[str] = []

    # Display a compact, fixed-height squad overview with elapsed time.
    iteration_label = state.get("iteration", 0)
    t0 = state.get("t0") or time.time()
    elapsed = time.time() - t0
    overview_lines = []
    for label, key in [("Squad Harmonic", "harmonic"), ("Squad Conflict", "conflict"), ("Squad Random", "random")]:
        names = [a.get("display_name") or a.get("name") for a in squad_assignments_raw.get(key, [])]
        overview_lines.append(f"{label}: {', '.join(names) if names else 'n/a'}")
    router.console.print(Text("=== Squad Overview ===", style="bold yellow"))
    for line in overview_lines:
        router.console.print(Text(line, style="cyan"))
    router.console.print(Text(f"Iteration: {iteration_label} | Elapsed: {elapsed:.1f}s", style="dim"))

    active_squads = [s for s in ["harmonic", "conflict", "random"] if squad_status.get(s, "active") != "done"]
    if not active_squads:
        router.console.print(Text("All squads completed. Skipping generation.", style="green"))
        return {
            **state,
            "generator_prompts": [],
            "squad_prompt_map": {},
            "history": history_entries,
            "prompt_squad_sequence": [],
        }

    for squad_name in active_squads:
        agent_entries = squad_assignments_raw.get(squad_name, [])
        contributions: List[Dict[str, str]] = []
        for agent_data in agent_entries:
            meta = _agent_from_dict(agent_data)
            persona = load_prompt(meta.prompt)
            system_prompt = f"{common_prompt}\n\n{persona}\nFollow the output format strictly."
            user_prompt = (
                f"Squad Type: {squad_name}\n"
                f"Base brief: {base_prompt}\n"
                f"Final Draft: {json.dumps(final_draft, ensure_ascii=False)}\n"
                f"Feedback from critics: {feedback or 'n/a'}\n"
                "React in character and propose SDXL-ready changes using the given format."
            )
            content = router.call(
                model=config.generator_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.9 if squad_name == "random" else 0.7,
                agent_label=f"{meta.display_name or meta.name} | Squad {squad_name.title()}",
            )
            contributions.append({"agent": meta.display_name or meta.name, "content": content})
            history_entries.append({"role": meta.display_name or meta.name, "content": content})

        prompt_text = _aggregate_squad_prompt(squad_name, final_draft, contributions, feedback, router, config)
        prompts_out.append(prompt_text)
        squad_prompt_map[squad_name] = prompt_text
        prompt_squad_sequence.append(squad_name)
        history_entries.append({"role": f"{squad_name.title()} Squad Prompt", "content": prompt_text})

    history = history_entries
    return {
        **state,
        "generator_prompts": prompts_out,
        "squad_prompt_map": squad_prompt_map,
        "prompt_squad_sequence": prompt_squad_sequence,
        "history": history,
    }


def critic_phase(state: MASState, config: SystemConfig, router: ModelRouter) -> MASState:
    router.console.print(Text(f"[phase:critic] iteration={state.get('iteration',0)}", style="bold white"))
    aliases = load_agent_aliases()
    directors = load_directors_auto(aliases)
    critic_common = load_prompt("prompts/directors/critic_mode/COMMON_INSTRUCTION_CRITIC.md")
    history_entries: List[Dict[str, str]] = state.get("history", []).copy()
    base_prompt = state.get("user_prompt", "")
    final_draft = state.get("final_draft", {})
    generator_prompts = state.get("generator_prompts", [])

    orchestrator_meta = next((d for d in directors if "orchestrator" in d.id), directors[-1])
    critic_metas = [d for d in directors if d.id != orchestrator_meta.id]

    results: List[Dict[str, Any]] = []
    prompt_squad_sequence = state.get("prompt_squad_sequence", []) or []
    images = state.get("images", []) or []
    squad_prompt_map = state.get("squad_prompt_map", {}) or {}
    squad_last_image: Dict[str, str] = state.get("squad_last_image", {}) or {}
    prev_squad_status = state.get("squad_status") or {"harmonic": "active", "conflict": "active", "random": "active"}

    # Ensure mapping length matches image count; if not, fall back to insertion order from squad_prompt_map.
    if len(prompt_squad_sequence) != len(images):
        fallback_seq = list(squad_prompt_map.keys())
        if not fallback_seq:
            fallback_seq = ["harmonic"] * len(images)
        prompt_squad_sequence = []
        for idx in range(len(images)):
            prompt_squad_sequence.append(fallback_seq[idx % len(fallback_seq)])

    for idx, img_path in enumerate(images):
        content_parts: List[Dict[str, Any]] = []
        text_body = (
            f"Image index: {idx+1}\n"
            f"Base brief: {base_prompt}\n"
            f"Final Draft: {json.dumps(final_draft, ensure_ascii=False)}\n"
            f"Prompt used: {generator_prompts[idx] if idx < len(generator_prompts) else ''}\n"
            "Evaluate strictly per your role."
        )
        content_parts.append({"type": "text", "text": text_body})
        p = Path(img_path)
        if p.exists():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            except OSError:
                pass

        critic_outputs: List[Dict[str, Any]] = []
        for meta in critic_metas:
            persona = load_prompt(meta.critic_prompt)
            system_prompt = f"{critic_common}\n\n{persona}\nOutput JSON only."
            content = router.call(
                model=config.director_model,
                system_prompt=system_prompt,
                user_prompt=text_body,
                temperature=0.4,
                agent_label=meta.display_name or meta.name,
                user_content=content_parts,
            )
            parsed = _safe_json_parse(content)
            critic_outputs.append({"agent": meta.display_name or meta.name, "data": parsed})
            history_entries.append({"role": meta.display_name or meta.name, "content": content})

        aggregate_input = "\n".join([json.dumps(c["data"], ensure_ascii=False) for c in critic_outputs])
        orch_persona = load_prompt(orchestrator_meta.critic_prompt)
        orch_system = f"{critic_common}\n\n{orch_persona}\nReturn JSON only."
        orch_user = (
            f"Critic JSON: {aggregate_input}\n"
            f"CI threshold: {config.creative_index_threshold}\n"
            "Compute CI, status, and refinement_instruction."
        )
        orch_raw = router.call(
            model=config.director_model,
            system_prompt=orch_system,
            user_prompt=orch_user,
            temperature=0.2,
            agent_label=orchestrator_meta.display_name or orchestrator_meta.name,
        )
        orch_parsed = _safe_json_parse(orch_raw)
        history_entries.append({"role": orchestrator_meta.display_name or orchestrator_meta.name, "content": orch_raw})
        results.append(
            {
                "image": img_path,
                "critics": critic_outputs,
                "orchestrator": orch_parsed,
            }
        )

    ci_scores: List[float] = []
    feedback_blocks: List[str] = []
    squad_ci: Dict[str, List[float]] = {"harmonic": [], "conflict": [], "random": []}
    for idx, item in enumerate(results):
        orch = item.get("orchestrator", {})
        ci_score = orch.get("ci_score", 0) if isinstance(orch, dict) else 0
        ci_scores.append(ci_score)
        squad_name = prompt_squad_sequence[idx] if idx < len(prompt_squad_sequence) else None
        if squad_name in squad_ci:
            squad_ci[squad_name].append(ci_score)
        refinement = orch.get("refinement_instruction") if isinstance(orch, dict) else None
        if refinement and isinstance(refinement, dict) and refinement.get("required"):
            bullets = refinement.get("bullet_points") or []
            summary = refinement.get("summary", "")
            block = f"Image {idx+1}: {summary}"
            if bullets:
                block += "\n- " + "\n- ".join(bullets)
            feedback_blocks.append(block)
    creative_index_avg = sum(ci_scores) / len(ci_scores) if ci_scores else 0.0
    squad_ci_avg = {s: (sum(vals) / len(vals) if vals else 0.0) for s, vals in squad_ci.items()}
    squad_status = state.get("squad_status") or {"harmonic": "active", "conflict": "active", "random": "active"}
    # track last image per squad from this iteration
    for idx, squad in enumerate(prompt_squad_sequence):
        if idx < len(images):
            squad_last_image[squad] = images[idx]
    for squad, avg in squad_ci_avg.items():
        if avg >= config.creative_index_threshold:
            squad_status[squad] = "done"
    # final images: latest per squad (order fixed)
    final_images: List[str] = []
    for sq in ["harmonic", "conflict", "random"]:
        if sq in squad_last_image:
            final_images.append(squad_last_image[sq])
    critic_feedback = "\n\n".join(feedback_blocks) if feedback_blocks else "Approved."

    history = history_entries
    return {
        **state,
        "critic_results": results,
        "creative_index_avg": creative_index_avg,
        "squad_ci_avg": squad_ci_avg,
        "squad_status": squad_status,
        "squad_last_image": squad_last_image,
        "final_images": final_images,
        "critic_feedback": critic_feedback,
        "history": history,
    }
