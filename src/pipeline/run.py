#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Thin wrapper to invoke the pipeline via the CLI as a subprocess.

This avoids duplicating the pipeline logic. It exposes a single function:
    run_pipeline(repo: str, out: Optional[Path]=None, overwrite=False,
                 skip_hf=False, skip_ailuminate=False, add_comment=False,
                 config: Optional[Path]=None, timeout: Optional[int]=None) -> int
which returns the process return code (0 on success).
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Optional


CLI_PATH = Path(__file__).resolve().parents[1] / "cli.py"


def run_pipeline(
    repo: str,
    out: Optional[Path] = None,
    overwrite: bool = False,
    skip_hf: bool = False,
    skip_ailuminate: bool = False,
    add_comment: bool = False,
    config: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> int:
    cmd = [
        sys.executable,
        str(CLI_PATH),
        "run",
        repo,
    ]
    if out is not None:
        cmd += ["-o", str(out)]
    if overwrite:
        cmd.append("--overwrite")
    if skip_hf:
        cmd.append("--skip-hf")
    if skip_ailuminate:
        cmd.append("--skip-ailuminate")
    if add_comment:
        cmd.append("--add-comment")
    if config is not None:
        cmd += ["--config", str(config)]
    if timeout is not None:
        cmd += ["--timeout", str(int(timeout))]

    proc = subprocess.run(cmd)
    return proc.returncode


__all__ = ["run_pipeline"]


