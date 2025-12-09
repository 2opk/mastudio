import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def collect_runs(globs: List[str]) -> List[Path]:
    runs: List[Path] = []
    for pattern in globs:
        for p in Path().glob(pattern):
            if p.is_dir():
                runs.append(p.resolve())
    return sorted(set(runs))


def load_report(run_dir: Path) -> Optional[Dict]:
    report = run_dir / "mas_report.json"
    if not report.exists():
        return None
    try:
        return json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_prompt(history: List[Dict[str, str]], role_key: str) -> Optional[str]:
    key = role_key.lower()
    for item in reversed(history):
        if item.get("role", "").lower() == key:
            return item.get("content", "")
    for item in reversed(history):
        if key in item.get("role", "").lower():
            return item.get("content", "")
    return None


def latest_image(intms_dir: Path, p_index: int) -> Optional[Tuple[Path, int]]:
    paths = sorted(intms_dir.glob(f"r*_p{p_index}_*.png"))
    if not paths:
        return None
    def iter_num(p: Path) -> int:
        stem = p.stem
        try:
            return int(stem.split("_")[0][1:])
        except Exception:
            return -1
    paths.sort(key=iter_num)
    last = paths[-1]
    return last.resolve(), iter_num(last)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_idx(run_name: str) -> Optional[int]:
    parts = run_name.split("_")
    if len(parts) >= 5:
        try:
            return int(parts[4])
        except Exception:
            return None
    return None


def copy_squad_images(run_dir: Path) -> None:
    intms = run_dir / "intms"
    if not intms.exists():
        return
    for pidx, label in [(1, "harmonic"), (2, "conflict"), (3, "random")]:
        res = latest_image(intms, pidx)
        if not res:
            continue
        img_path, _ = res
        dst = run_dir / f"squad_{label}.png"
        shutil.copy2(img_path, dst)


def harvest(runs: List[Path], post_dir: Path, template_path: Path) -> None:
    _reset_dir(post_dir)
    buckets = {
        "all": (post_dir / "all_images", post_dir / "all_prompts.json"),
        "harmonic": (post_dir / "harmonic_images", post_dir / "harmonic_prompts.json"),
        "conflict": (post_dir / "conflict_images", post_dir / "conflict_prompts.json"),
        "random": (post_dir / "random_images", post_dir / "random_prompts.json"),
    }
    for dir_path, _ in buckets.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    template_data = json.loads(template_path.read_text(encoding="utf-8"))
    prompt_maps: Dict[str, Dict[str, str]] = {k: {} for k in buckets.keys()}

    for run_dir in runs:
        data = load_report(run_dir)
        if not data:
            continue
        final_state = data.get("final_state", {})
        history = final_state.get("history") or data.get("history") or []
        if not isinstance(history, list):
            history = []
        idx = parse_idx(run_dir.name)

        copy_squad_images(run_dir)

        mapping = {
            "harmonic": ("harmonic squad prompt", 1),
            "conflict": ("conflict squad prompt", 2),
            "random": ("random squad prompt", 3),
        }
        squad_prompts: Dict[str, str] = {}
        squad_images: Dict[str, Tuple[Path, int]] = {}
        intms = run_dir / "intms"
        for squad, (role, pidx) in mapping.items():
            prompt = latest_prompt(history, role)
            res = latest_image(intms, pidx)
            if prompt:
                squad_prompts[squad] = prompt
            if res:
                squad_images[squad] = res

        if idx is not None:
            for entry in template_data:
                if entry.get("sample_idx") == idx:
                    pid = entry.get("prompt_id")
                    if pid == 0 and "harmonic" in squad_prompts:
                        entry["visual_prompt"] = squad_prompts["harmonic"]
                    elif pid == 1 and "conflict" in squad_prompts:
                        entry["visual_prompt"] = squad_prompts["conflict"]
                    elif pid == 2 and "random" in squad_prompts:
                        entry["visual_prompt"] = squad_prompts["random"]

        run_id = run_dir.name
        for squad, pidx in [("harmonic", 1), ("conflict", 2), ("random", 3)]:
            if squad not in squad_prompts or squad not in squad_images:
                continue
            img_path, iter_num = squad_images[squad]
            dst_name = f"image_{run_id}_p{pidx}_r{iter_num}.png"
            squad_dir, _ = buckets[squad]
            shutil.copy2(img_path, squad_dir / dst_name)
            prompt_maps[squad][dst_name] = squad_prompts[squad]
            all_dir, _ = buckets["all"]
            shutil.copy2(img_path, all_dir / dst_name)
            prompt_maps["all"][dst_name] = squad_prompts[squad]

    for key, (_, json_path) in buckets.items():
        json_path.write_text(json.dumps(prompt_maps[key], ensure_ascii=False, indent=2), encoding="utf-8")
    out_template = post_dir / "prompts_for_prompt_creativity.json"
    out_template.write_text(json.dumps(template_data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Postprocess runs: split images/prompts into buckets and fill prompt creativity JSON.")
    parser.add_argument("globs", nargs="+", help="Run directory globs, e.g., output_qwen/output_20251208_*")
    args = parser.parse_args()

    runs = collect_runs(args.globs)
    if not runs:
        print("No run directories found.")
        return
    base_dir = runs[0].parent
    post_dir = base_dir / "postprocessed"
    template_path = Path("metric_calculation/prompt_creativity/prompts_for_image_generation_template.json")
    if not template_path.exists():
        src = Path("metric_calculation/prompt_creativity/prompts_for_image_generation.json")
        if src.exists():
            data = json.loads(src.read_text(encoding="utf-8"))
            filtered = []
            for item in data:
                if item.get("prompt_id", 99) > 2:
                    continue
                new_item = dict(item)
                new_item["visual_prompt"] = ""
                filtered.append(new_item)
            template_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print("Template file missing.")
            return

    harvest(runs, post_dir, template_path)
    print(f"Postprocessing complete -> {post_dir}")


if __name__ == "__main__":
    main()
