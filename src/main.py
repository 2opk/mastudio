import argparse
import re
import shutil
import time
from pathlib import Path
from typing import Dict

from .graph import build_app
from .tools import save_json_report
from .utils import load_env, load_system_config, prepare_run_dirs


_PROMPT_IMAGE_RE = re.compile(r"r(\d+)_p(\d+)_", re.IGNORECASE)


def _latest_prompt_images(intms_dir: Path) -> Dict[int, Path]:
    """
    Return mapping: prompt_index (1-based) -> latest image path across iterations.
    """
    latest: Dict[int, tuple[int, Path]] = {}
    for path in intms_dir.glob("r*_p*.png"):
        match = _PROMPT_IMAGE_RE.match(path.stem)
        if not match:
            continue
        iteration = int(match.group(1))
        prompt_idx = int(match.group(2))
        prev = latest.get(prompt_idx)
        if prev is None or iteration >= prev[0]:
            latest[prompt_idx] = (iteration, path)
    return {idx: p for idx, (_, p) in latest.items()}


def run(base_prompt: str, max_loops: int = None, output_dir: str = None, config_path: str = None):
    load_env()
    config = load_system_config(config_path or "config/system_config.yaml")
    if max_loops is not None:
        config.max_loops = max_loops
    # Reset and prepare run directories.
    run_root = output_dir or config.sdxl.output_dir
    use_as_run_dir = bool(output_dir)
    run_dir, intms_dir, timestamp = prepare_run_dirs(run_root, use_as_run_dir=use_as_run_dir)
    # Point SDXL outputs to intermediates folder for this run.
    config.sdxl.output_dir = str(intms_dir)

    app = build_app(config)
    initial_state = {"user_prompt": base_prompt, "iteration": 0, "history": [], "t0": time.time()}
    result = app.invoke(initial_state, config={"recursion_limit": config.recursion_limit})

    # Copy final images from intms to run root.
    final_images = result.get("final_images") or result.get("images", [])
    final_images_out = []
    # Copy primary outputs (p1/p2/p3 order).
    for image_path in final_images:
        src = Path(image_path)
        if not src.exists():
            continue
        dest = run_dir / src.name
        shutil.copy2(src, dest)
        final_images_out.append(str(dest))

    # Also copy squad-specific last images if available.
    squad_last = result.get("squad_last_image", {}) or {}
    for squad, img_path in squad_last.items():
        src = Path(img_path)
        if not src.exists():
            continue
        dest = run_dir / f"squad_{squad}.png"
        shutil.copy2(src, dest)
        final_images_out.append(str(dest))

    # Ensure we keep the latest image per prompt index (p1/p2/p3) as squad outputs.
    latest_by_prompt = _latest_prompt_images(intms_dir)
    idx_to_squad = {1: "harmonic", 2: "conflict", 3: "random"}
    for idx, src in latest_by_prompt.items():
        squad = idx_to_squad.get(idx)
        if not squad or not src.exists():
            continue
        dest_names = [f"squad_{squad}.png"]
        if squad == "harmonic":
            dest_names.append("squad_harmony.png")  # backward-compatible alias
        for dest_name in dest_names:
            dest = run_dir / dest_name
            shutil.copy2(src, dest)
            final_images_out.append(str(dest))

    # Deduplicate while preserving order.
    final_images_out = list(dict.fromkeys(final_images_out))

    report_path = run_dir / "mas_report.json"
    save_json_report(
        {
            "base_prompt": base_prompt,
            "timestamp": timestamp,
            "run_dir": str(run_dir),
            "intermediates_dir": str(intms_dir),
            "final_images": final_images_out,
            "final_state": result,
        },
        report_path,
    )
    return result, report_path


def main():
    parser = argparse.ArgumentParser(description="Run Director–Generator MAS for generative art.")
    parser.add_argument("prompt", nargs="?", help="Base text prompt describing music/mood. If omitted, use -f/--prompt-file.")
    parser.add_argument(
        "-f",
        "--prompt-file",
        dest="prompt_file",
        help="Path to a text file containing the base prompt; file contents will be used verbatim.",
    )
    parser.add_argument(
        "--max-iterations",
        "--max-iteration",
        type=int,
        dest="max_iterations",
        help="Override max loop iterations (default comes from config).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Root directory for outputs (will create timestamped run dir inside).",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to system config YAML (default: config/system_config.yaml).",
    )
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    elif args.prompt:
        base_prompt = args.prompt
    else:
        parser.error("Provide a prompt string or a file via -f/--prompt-file.")

    result, report_path = run(
        base_prompt,
        max_loops=args.max_iterations,
        output_dir=args.output_dir,
        config_path=args.config_path,
    )
    print(f"Run complete. Final iteration: {result.get('iteration')}. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
