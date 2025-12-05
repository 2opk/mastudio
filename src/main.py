import argparse
import shutil
from pathlib import Path

from .graph import build_app
from .tools import save_json_report
from .utils import load_env, load_system_config, prepare_run_dirs


def run(base_prompt: str):
    load_env()
    config = load_system_config()
    # Reset and prepare run directories.
    run_dir, intms_dir, timestamp = prepare_run_dirs(config.sdxl.output_dir)
    # Point SDXL outputs to intermediates folder for this run.
    config.sdxl.output_dir = str(intms_dir)

    app = build_app(config)
    initial_state = {"user_prompt": base_prompt, "iteration": 0, "history": []}
    result = app.invoke(initial_state, config={"recursion_limit": config.recursion_limit})

    # Copy final images from intms to run root.
    final_images_out = []
    for image_path in result.get("images", []):
        src = Path(image_path)
        if not src.exists():
            continue
        dest = run_dir / src.name
        shutil.copy2(src, dest)
        final_images_out.append(str(dest))

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
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    elif args.prompt:
        base_prompt = args.prompt
    else:
        parser.error("Provide a prompt string or a file via -f/--prompt-file.")

    result, report_path = run(base_prompt)
    print(f"Run complete. Final iteration: {result.get('iteration')}. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
