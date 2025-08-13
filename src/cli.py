#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified entry CLI for AI-BOM tooling.

Commands:
  - gen: generate AI-BOM (wraps src/hf2spdx_ai_bom.py)
  - enrich hf: enrich HF fields (wraps src/enrichers/spdx_hf_enricher.py)
  - enrich ailuminate: enrich AILuminate risk (wraps src/enrichers/spdx_ailuminate_enricher.py)
  - run: pipeline: gen -> enrich hf -> enrich ailuminate

Notes:
  - Only standard library is used here.
  - All underlying calls use subprocess with sys.executable to respect the active venv.
  - Flags are passed through conservatively: only to scripts that support them.
  - On failure, this CLI returns the same returncode and prints a brief child-output summary to stderr.
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shlex


ROOT = Path(__file__).resolve().parent
SCRIPT_GEN = ROOT / "hf2spdx_ai_bom.py"
SCRIPT_ENRICH_HF = ROOT / "enrichers" / "spdx_hf_enricher.py"
SCRIPT_ENRICH_AIL = ROOT / "enrichers" / "spdx_ailuminate_enricher.py"


def _tail_text(text: str, limit_lines: int = 50) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= limit_lines:
        return text or ""
    tail = "\n".join(lines[-limit_lines:])
    return tail


def _run_child(cmd: list[str], timeout: int | None = None) -> int:
    """Run a child command. Print stdout on success. On failure, print tail of both streams to stderr.
    Return the child's return code.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # Synthesize a concise error
        err = f"[cli] Subprocess timed out after {timeout}s: {shlex.join(cmd)}\n"
        err += _tail_text((e.stdout or "") + "\n" + (e.stderr or ""))
        sys.stderr.write(err + "\n")
        return 124  # conventional timeout code

    if proc.returncode == 0:
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            # Preserve warnings/info from child
            sys.stderr.write(proc.stderr)
        return 0

    summary = [
        f"[cli] Subprocess failed (code {proc.returncode}): {shlex.join(cmd)}",
        _tail_text((proc.stdout or "").strip()),
        _tail_text((proc.stderr or "").strip()),
    ]
    sys.stderr.write("\n".join([s for s in summary if s]) + "\n")
    return proc.returncode


def _safe_repo_id(input_str: str) -> str:
    # Accept either repo_id or HF URL; return repo_id-like owner/repo
    if input_str.startswith(("http://", "https://")) and "huggingface.co" in input_str:
        try:
            # Minimal parser to avoid extra deps
            path = input_str.split("huggingface.co", 1)[1]
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        except Exception:
            pass
    return input_str


def _default_output_path(repo_id_or_url: str) -> Path:
    rid = _safe_repo_id(repo_id_or_url)
    safe = rid.replace("/", "_")
    date = datetime.utcnow().strftime("%Y%m%d")
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe}_{date}.spdx3.json"


def cmd_gen(args: argparse.Namespace) -> int:
    cmd = [str(SCRIPT_GEN), args.repo]
    if args.out is not None:
        cmd += ["-o", str(args.out)]
    # passthrough: --timeout only (gen does not support --dry-run)
    if args.timeout is not None:
        cmd += ["--timeout", str(int(args.timeout))]
    return _run_child([sys.executable, *cmd], timeout=args.timeout)


def cmd_enrich_hf(args: argparse.Namespace) -> int:
    cmd = [str(SCRIPT_ENRICH_HF), *[str(p) for p in args.inputs]]
    if args.out is not None:
        cmd += ["-o"]
        if args.out != "":
            cmd += [str(args.out)]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.timeout is not None:
        cmd += ["--timeout", str(int(args.timeout))]
    return _run_child([sys.executable, *cmd], timeout=args.timeout)


def cmd_enrich_ailuminate(args: argparse.Namespace) -> int:
    cmd = [str(SCRIPT_ENRICH_AIL), *[str(p) for p in args.inputs]]
    if args.out is not None:
        cmd += ["-o"]
        if args.out != "":
            cmd += [str(args.out)]
    if args.add_comment:
        cmd += ["--add-comment"]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.timeout is not None:
        cmd += ["--timeout", str(int(args.timeout))]
    return _run_child([sys.executable, *cmd], timeout=args.timeout)


def _read_simple_yaml(path: Path) -> dict:
    """Minimal YAML parser for flat key: value pairs (bool/int/str). Comments (#) and empty lines ignored.
    If parsing fails, return {}.
    """
    data = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        # strip quotes if present
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        low = val.lower()
        if low in {"true", "false"}:
            data[key] = (low == "true")
        else:
            try:
                data[key] = int(val)
            except Exception:
                data[key] = val
    return data


def cmd_run(args: argparse.Namespace) -> int:
    # Merge config defaults (optional) with CLI flags; CLI wins.
    cfg = {}
    if args.config is not None:
        cfg = _read_simple_yaml(Path(args.config)) or {}
    else:
        # Try default config path if exists
        default_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        if default_cfg_path.exists():
            cfg = _read_simple_yaml(default_cfg_path) or {}

    skip_hf = bool(args.skip_hf if args.skip_hf is not None else cfg.get("skip_hf", False))
    skip_ail = bool(args.skip_ailuminate if args.skip_ailuminate is not None else cfg.get("skip_ailuminate", False))
    overwrite = bool(args.overwrite if args.overwrite is not None else cfg.get("overwrite", False))
    add_comment = bool(args.add_comment if args.add_comment is not None else cfg.get("add_comment", False))

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        # allow config override of output_dir
        out_dir = Path(cfg.get("output_dir", "output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = _default_output_path(args.repo)
        if out_dir != Path("output"):
            out_path = out_dir / out_path.name

    # If output exists and not overwrite -> abort early
    if out_path.exists() and not overwrite:
        sys.stderr.write(f"[cli] Output already exists, use --overwrite to replace: {out_path}\n")
        return 3

    # 1) gen
    sys.stdout.write("[cli] Step 1/3: gen...\n")
    rc = _run_child([
        sys.executable,
        str(SCRIPT_GEN),
        args.repo,
        "-o", str(out_path),
        *(["--timeout", str(int(args.timeout))] if args.timeout is not None else []),
    ], timeout=args.timeout)
    if rc != 0:
        return rc

    # 2) enrich hf
    if not skip_hf:
        sys.stdout.write("[cli] Step 2/3: enrich hf...\n")
        rc = _run_child([
            sys.executable,
            str(SCRIPT_ENRICH_HF),
            str(out_path),
            "-o", "orig",  # in-place
            *(["--timeout", str(int(args.timeout))] if args.timeout is not None else []),
        ], timeout=args.timeout)
        if rc != 0:
            return rc
    else:
        sys.stdout.write("[cli] Step 2/3: enrich hf skipped.\n")

    # 3) enrich ailuminate
    if not skip_ail:
        sys.stdout.write("[cli] Step 3/3: enrich ailuminate...\n")
        base_cmd = [
            sys.executable,
            str(SCRIPT_ENRICH_AIL),
            str(out_path),
            "-o", "orig",
        ]
        if add_comment:
            base_cmd.append("--add-comment")
        if args.timeout is not None:
            base_cmd += ["--timeout", str(int(args.timeout))]
        rc = _run_child(base_cmd, timeout=args.timeout)
        if rc != 0:
            return rc
    else:
        sys.stdout.write("[cli] Step 3/3: enrich ailuminate skipped.\n")

    sys.stdout.write(f"[cli] Pipeline success. Output: {out_path}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified CLI for AI-BOM generation and enrichment")
    sub = p.add_subparsers(dest="command", required=True)

    # gen
    pg = sub.add_parser("gen", help="產生 AI-BOM（呼叫 src/hf2spdx_ai_bom.py）")
    pg.add_argument("repo", help="Hugging Face repo_id 或 URL")
    pg.add_argument("-o", "--out", help="輸出檔案路徑（傳遞為 -o/--output）")
    pg.add_argument("--timeout", type=int, help="HTTP timeout 秒數（透傳）")
    pg.set_defaults(func=cmd_gen)

    # enrich group
    pe = sub.add_parser("enrich", help="補齊欄位")
    se = pe.add_subparsers(dest="enrich_target", required=True)

    peh = se.add_parser("hf", help="補齊 HF 欄位（呼叫 src/enrichers/spdx_hf_enricher.py）")
    peh.add_argument("inputs", nargs="+", help="1..N 個 AI-BOM JSON 檔案")
    peh.add_argument("-o", "--out", nargs="?", const="", help="orig|inplace|overwrite 或 <filename>；無值時為 per-file enriched.<name>")
    peh.add_argument("--dry-run", action="store_true", help="只列印，不寫入（透傳）")
    peh.add_argument("--timeout", type=int, help="HTTP timeout 秒數（透傳）")
    peh.set_defaults(func=cmd_enrich_hf)

    pea = se.add_parser("ailuminate", help="補齊 AILuminate 風險（呼叫 src/enrichers/spdx_ailuminate_enricher.py）")
    pea.add_argument("inputs", nargs="+", help="1..N 個 AI-BOM JSON 檔案")
    pea.add_argument("-o", "--out", nargs="?", const="", help="orig|inplace|overwrite 或 <filename>；無值時為 per-file enriched.<name>")
    pea.add_argument("--add-comment", action="store_true", help="寫入來源說明到 AIPackage.comment（透傳）")
    pea.add_argument("--dry-run", action="store_true", help="只列印，不寫入（透傳）")
    pea.add_argument("--timeout", type=int, help="HTTP timeout 秒數（透傳）")
    pea.set_defaults(func=cmd_enrich_ailuminate)

    # run pipeline
    pr = sub.add_parser("run", help="一鍵管線：gen → enrich hf → enrich ailuminate")
    pr.add_argument("repo", help="Hugging Face repo_id 或 URL")
    pr.add_argument("-o", "--out", help="最終輸出檔案路徑；未指定時預設為 output/<safe_model_id>_<YYYYMMDD>.spdx3.json")
    pr.add_argument("--overwrite", action="store_true", help="當輸出已存在時覆寫")
    pr.add_argument("--skip-hf", action="store_true", help="略過 HF 補強步驟")
    pr.add_argument("--skip-ailuminate", action="store_true", help="略過 AILuminate 補強步驟")
    pr.add_argument("--add-comment", action="store_true", help="在 AILuminate 步驟加入來源說明")
    pr.add_argument("--config", help="讀取設定檔（預設為 configs/default.yaml，如不存在則忽略）")
    pr.add_argument("--timeout", type=int, help="HTTP timeout 秒數（透傳至所有步驟並作為子程序逾時計時）")
    pr.set_defaults(func=cmd_run)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()


