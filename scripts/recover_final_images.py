#!/usr/bin/env python3
"""
Recover final squad images for past runs where early-exit squads were not copied.

Assumptions:
- Images live in <run_dir>/intms/ named r{iter}_p{idx}.png (idx is 1-based order of active squads).
- Generator logs include lines like "[phase:generator] iteration=X" and agent lines
  "[<name> | Squad Harmonic]" etc. We use these to map p{idx} -> squad per iteration.
Usage:
  python scripts/recover_final_images.py output_20251208_*  # one or more run dirs
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


SQUAD_PATTERN = re.compile(r"\[.+\|\s*Squad\s+(Harmonic|Conflict|Random)\]", re.IGNORECASE)
ITER_PATTERN = re.compile(r"\[phase:generator\]\s+iteration=(\d+)", re.IGNORECASE)
SQ_OVERVIEW_PATTERN = re.compile(r"Squad\s+(Harmonic|Conflict|Random):", re.IGNORECASE)


def parse_log(log_path: Path) -> Dict[int, List[str]]:
    """
    Return mapping: iteration -> list of squads in order seen.
    Use the first Squad Overview block per iteration if present, otherwise
    fall back to chat lines. If still missing, leave entry empty for fallback.
    """
    mapping: Dict[int, List[str]] = {}
    current_iter = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_iter = ITER_PATTERN.search(line)
            if m_iter:
                current_iter = int(m_iter.group(1))
                mapping[current_iter] = []
                continue
            if current_iter is None:
                continue
            # prefer overview order if we haven't captured squads for this iteration yet
            if not mapping[current_iter]:
                if "Squad Harmonic" in line:
                    mapping[current_iter].append("harmonic")
                    continue
                if "Squad Conflict" in line:
                    mapping[current_iter].append("conflict")
                    continue
                if "Squad Random" in line:
                    mapping[current_iter].append("random")
                    continue
            m_sq = SQUAD_PATTERN.search(line)
            if m_sq:
                squad = m_sq.group(1).lower()
                if squad not in mapping[current_iter]:
                    mapping[current_iter].append(squad)
    return mapping


def recover_from_report(run_dir: Path) -> Optional[Dict[str, str]]:
    report = run_dir / "mas_report.json"
    if not report.exists():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        final_state = data.get("final_state", {})
        squad_last = final_state.get("squad_last_image") or final_state.get("squad_last") or {}
        if squad_last:
            return {str(k): str(v) for k, v in squad_last.items()}
    except Exception:
        return None
    return None


def recover(run_dir: Path) -> None:
    intms = run_dir / "intms"
    log_path = run_dir / "run.log"
    if not intms.exists() or not log_path.exists():
        print(f"[skip] {run_dir} missing intms or run.log")
        return
    iter_squads = parse_log(log_path)
    # Prefer report if available.
    squad_last_report = recover_from_report(run_dir)
    squad_last: Dict[str, Path] = {}
    if squad_last_report:
        for squad, p in squad_last_report.items():
            path = Path(p)
            if not path.is_absolute():
                path = run_dir / path
            if path.exists():
                squad_last[squad] = path

    # Fallback: infer from intms + log
    if not squad_last:
        squad_last_iter: Dict[str, int] = {}
        idx_last: Dict[int, Path] = {}
        idx_iter: Dict[int, int] = {}
        for img_path in sorted(intms.glob("r*_p*.png")):
            stem = img_path.stem  # e.g., r2_p1_xxx
            parts = stem.split("_")
            try:
                iter_num = int(parts[0][1:])
                prompt_idx = int(parts[1][1:]) - 1  # 0-based
            except Exception:
                continue
            squads = iter_squads.get(iter_num) or []
            default_order = ["harmonic", "conflict", "random"]
            if (not squads) or (len(squads) < len(default_order)) or (prompt_idx >= len(squads)):
                default_order = ["harmonic", "conflict", "random"]
                if prompt_idx < len(default_order):
                    squad = default_order[prompt_idx]
                else:
                    # still track by index for fallback
                    if iter_num >= idx_iter.get(prompt_idx, -1):
                        idx_iter[prompt_idx] = iter_num
                        idx_last[prompt_idx] = img_path
                    continue
            else:
                squad = squads[prompt_idx]
            prev_iter = squad_last_iter.get(squad, -1)
            if iter_num >= prev_iter:
                squad_last_iter[squad] = iter_num
                squad_last[squad] = img_path
        # fill missing squads using index-based latest
        order = ["harmonic", "conflict", "random"]
        for idx, squad in enumerate(order):
            if squad not in squad_last and idx in idx_last:
                squad_last[squad] = idx_last[idx]

    if not squad_last:
        print(f"[warn] No squad mapping recovered for {run_dir}")
        return
    for squad, img in squad_last.items():
        dest = run_dir / f"squad_{squad}.png"
        dest.write_bytes(img.read_bytes())
        print(f"[ok] {run_dir.name}: {squad} -> {dest.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/recover_final_images.py <run_dir> [run_dir ...]")
        sys.exit(1)
    for path_str in sys.argv[1:]:
        recover(Path(path_str))


if __name__ == "__main__":
    main()
