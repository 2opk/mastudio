import base64
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union
import time

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
    stable_seed,
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
    critic_feedback: str
    t0: float


class ModelRouter:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.console = Console()
        self._logged_models = set()

    def _preview(self, text: Any, limit: int = 1000) -> str:
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

    def _sanitize_preview(self, raw: str, limit: int) -> str:
        s = raw.replace("\\n", "\n").replace('\\"', '"')
        return s if len(s) <= limit else s[: limit - 3] + "..."

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
        header = Text(f"[{agent_label}] {direction}", style="bold cyan" if direction == "→" else "bold magenta")
        body = Text(f"{text}", style="grey70")
        self.console.print(header)
        self.console.print(body)
        self.console.print()  # blank line for readability


def _safe_json_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw.strip()}


def _normalize_keywords(keywords: List[str]) -> List[str]:
    normalized: List[str] = []
    for kw in keywords:
        if not kw:
            continue
        clean = kw.lower().strip()
        if not clean:
            continue
        normalized.append(clean)
        normalized.extend([token for token in clean.split() if len(token) > 2])
    return normalized


def _text_keywords(text: str, limit: int = 12) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    filtered: List[str] = []
    for token in tokens:
        if len(filtered) >= limit:
            break
        if len(token) < 3:
            continue
        filtered.append(token)
    return filtered


def _score_agent(agent: GeneratorAgentMeta, targets: List[str]) -> int:
    if not targets:
        return 0
    agent_kw = _normalize_keywords(agent.keywords)
    if not agent_kw:
        return 0
    score = 0
    for t in targets:
        for ak in agent_kw:
            if t in ak or ak in t:
                score += 1
                break
    return score


def _select_by_score(pool: List[GeneratorAgentMeta], targets: List[str], reverse: bool = True, k: int = 3) -> List[GeneratorAgentMeta]:
    scored = sorted(pool, key=lambda a: _score_agent(a, targets), reverse=reverse)
    return scored[:k] if scored else []


def _fallback_random(pool: List[GeneratorAgentMeta], seed_text: str, k: int = 3) -> List[GeneratorAgentMeta]:
    rng = random.Random(stable_seed(seed_text))
    shuffled = pool.copy()
    rng.shuffle(shuffled)
    return shuffled[:k]


def select_squad_agents(
    base_prompt: str,
    final_draft: Dict[str, Any],
    instructions: Dict[str, Any],
    pool: List[GeneratorAgentMeta],
) -> Dict[str, List[GeneratorAgentMeta]]:
    main_prompt = ""
    if isinstance(final_draft, dict):
        main_prompt = final_draft.get("main_prompt", "") or json.dumps(final_draft, ensure_ascii=False)
    base_targets = _text_keywords(main_prompt or base_prompt)
    harmonic_targets = instructions.get("squad_a_harmonic", {}).get("target_keywords") or base_targets
    conflict_targets = instructions.get("squad_b_conflict", {}).get("target_keywords") or base_targets
    categories = [
        "category1_time",
        "category2_cognitive",
        "category3_emotional",
        "category4_cultural",
        "category5_medium",
    ]
    grouped: Dict[str, List[GeneratorAgentMeta]] = defaultdict(list)
    for agent in pool:
        grouped[agent.category or "uncategorized"].append(agent)

    def _pick_by_strategy(cat: str, targets: List[str], reverse: bool, seed_suffix: str) -> Optional[GeneratorAgentMeta]:
        agents = grouped.get(cat, [])
        if not agents:
            return None
        norm_targets = _normalize_keywords(targets)
        if not norm_targets:
            rng = random.Random(stable_seed(base_prompt + main_prompt + seed_suffix + cat))
            return rng.choice(agents)
        scored = sorted(
            agents,
            key=lambda a: (_score_agent(a, norm_targets), a.id),
            reverse=reverse,
        )
        return scored[0] if scored else None

    harmonic_agents: List[GeneratorAgentMeta] = []
    conflict_agents: List[GeneratorAgentMeta] = []
    random_agents: List[GeneratorAgentMeta] = []

    for cat in categories:
        chosen_h = _pick_by_strategy(cat, harmonic_targets, True, "harmonic") or _fallback_random(
            pool, base_prompt + "harmonic" + cat, k=1
        )[0]
        chosen_c = _pick_by_strategy(cat, conflict_targets, False, "conflict") or _fallback_random(
            pool, base_prompt + "conflict" + cat, k=1
        )[0]
        rng = random.Random(stable_seed(base_prompt + main_prompt + "random" + cat))
        cat_agents = grouped.get(cat) or pool
        chosen_r = rng.choice(cat_agents)
        harmonic_agents.append(chosen_h)
        conflict_agents.append(chosen_c)
        random_agents.append(chosen_r)

    return {
        "harmonic": harmonic_agents,
        "conflict": conflict_agents,
        "random": random_agents,
    }


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
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip(",;") + "…"
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
        "Synthesize into final_draft and squad_selection_instructions JSON."
    )
    orchestrator_raw = _call_director(directors[3], orch_payload)
    orchestrator_json = _safe_json_parse(orchestrator_raw)
    history_entries.append({"role": directors[3].display_name or directors[3].name, "content": orchestrator_raw})

    final_draft = orchestrator_json.get("final_draft", {}) if isinstance(orchestrator_json, dict) else {}
    squad_instructions = orchestrator_json.get("squad_selection_instructions", {}) if isinstance(orchestrator_json, dict) else {}

    pool = load_generator_pool()
    squad_assignments = select_squad_agents(base_prompt, final_draft, squad_instructions, pool)

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

    return {
        **state,
        "mood_report": mi_output,
        "visual_blueprint": vd_output,
        "conceptual_elements": ca_output,
        "final_draft": final_draft,
        "squad_selection_instructions": squad_instructions,
        "squad_assignments": squad_assignments_dict,
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
        "You are the Squad Synthesizer. Merge the dialogue into ONE SDXL-ready prompt, <=77 characters. "
        "Preserve Final Draft intent, include camera/lighting/color/style, and bake in feedback."
        "Return ONLY the prompt as plain text (no JSON, no labels, no code fences)."
    )
    convo_text = "\n\n".join([f"{c['agent']}: {c['content']}" for c in contributions])
    user_prompt = (
        f"Squad: {squad_name}\n"
        f"Final Draft: {json.dumps(final_draft, ensure_ascii=False)}\n"
        f"Feedback to apply: {feedback}\n"
        f"Dialogue:\n{convo_text}\n"
        "Return one plain-text prompt <=77 chars."
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

    squad_assignments_raw = state.get("squad_assignments") or {}
    if not squad_assignments_raw:
        # Safety fallback if state was missing assignments.
        pool = load_generator_pool()
        instructions = state.get("squad_selection_instructions", {})
        squad_assignments = select_squad_agents(base_prompt, final_draft, instructions, pool)
        squad_assignments_raw = {
            squad: [
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
            for squad, metas in squad_assignments.items()
        }

    prompts_out: List[str] = []
    squad_prompt_map: Dict[str, str] = {}

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

    for squad_name in ["harmonic", "conflict", "random"]:
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
        history_entries.append({"role": f"{squad_name.title()} Squad Prompt", "content": prompt_text})

    history = history_entries
    return {
        **state,
        "generator_prompts": prompts_out,
        "squad_prompt_map": squad_prompt_map,
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

    for idx, img_path in enumerate(state.get("images", [])):
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
    for idx, item in enumerate(results):
        orch = item.get("orchestrator", {})
        ci_score = orch.get("ci_score", 0) if isinstance(orch, dict) else 0
        ci_scores.append(ci_score)
        refinement = orch.get("refinement_instruction") if isinstance(orch, dict) else None
        if refinement and isinstance(refinement, dict) and refinement.get("required"):
            bullets = refinement.get("bullet_points") or []
            summary = refinement.get("summary", "")
            block = f"Image {idx+1}: {summary}"
            if bullets:
                block += "\n- " + "\n- ".join(bullets)
            feedback_blocks.append(block)
    creative_index_avg = sum(ci_scores) / len(ci_scores) if ci_scores else 0.0
    critic_feedback = "\n\n".join(feedback_blocks) if feedback_blocks else "Approved."

    history = history_entries
    return {
        **state,
        "critic_results": results,
        "creative_index_avg": creative_index_avg,
        "critic_feedback": critic_feedback,
        "history": history,
    }
