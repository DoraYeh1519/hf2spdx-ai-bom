#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

AIL_PAGE = "https://ailuminate.mlcommons.org/benchmarks/general_purpose_ai_chat/1.0-en_us-official-ensemble"

GRADE_TO_RISK = {
    "excellent": "low",
    "very good": "low",
    "good": "medium",
    "fair": "high",
    "poor": "serious",
}

# Common vendor/platform tokens that we ignore in matching
STOP_TOKENS = {
    "hf","together","api","azure","openai","meta","google","ai21labs","cohere","mistralai",
    "minstral","gemini","deepseek","nvidia","nebius","anthropic","amazon","aws","gpt","chat",
    "model","inference","turbo","ultra","sonnet","haiku","flash","lite","with","moderation",
    "recipe","command","large"
}

def normalize_tokens(name: str) -> List[str]:
    """
    Normalize a model name into tokens for conservative matching.
    - Lowercase
    - Replace separators [_\-/.] with spaces
    - Collapse number like '3.1' and '3 1' to '31' tokens
    - Keep letter-number tokens like '8b' together
    - Remove vendor/platform stop-words
    - Return a sorted unique token list
    """
    s = name.lower()
    s = re.sub(r"[()/@]", " ", s)
    s = re.sub(r"[_\-\.\,]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    raw = s.split()

    toks: List[str] = []
    for t in raw:
        if t in STOP_TOKENS:
            continue
        # join dotted/space-separated versions like '3 1' -> '31'
        t = re.sub(r"[^a-z0-9]+", "", t)
        if not t:
            continue
        toks.append(t)
    # unique and sort by length then alpha to stabilize comparisons
    uniq = sorted(set(toks), key=lambda x: (len(x), x))
    return uniq

def hf_repo_from_aipackage(doc: Dict[str, Any]) -> Optional[str]:
    for elem in doc.get("element", []):
        if isinstance(elem, dict) and elem.get("type") == "AIPackage":
            name = elem.get("name")
            if isinstance(name, str) and "/" in name:
                return name
            dl = elem.get("downloadLocation")
            if isinstance(dl, str):
                try:
                    u = urlparse(dl)
                    if "huggingface.co" in (u.netloc or ""):
                        parts = [p for p in u.path.strip("/").split("/") if p]
                        if len(parts) >= 2:
                            return f"{parts[0]}/{parts[1]}"
                except Exception:
                    pass
    return None

def core_repo_tokens(repo_id: str) -> List[str]:
    # Only the repo "name" part after owner/
    try:
        name = repo_id.split("/", 1)[1]
    except Exception:
        name = repo_id
    return normalize_tokens(name)

def fetch_ailuminate_bare_models(timeout: int=30) -> List[Tuple[str, str, Optional[str]]]:
    """
    Returns list of (model_name, grade, details_url) from AILuminate Bare Models.
    """
    r = requests.get(AIL_PAGE, timeout=timeout)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    # Locate Bare Models section by h2/h3 heading text
    bare_header = None
    for h in soup.find_all(re.compile("^h[1-6]$")):
        if h.get_text(strip=True).lower().startswith("bare models"):
            bare_header = h
            break
    if not bare_header:
        # fallback: regex scan text
        text = soup.get_text("\n", strip=True)
        return _regex_extract_bare_models(text)

    # Collect entries that visually appear between this heading and next heading
    entries: List[Tuple[str,str,Optional[str]]] = []
    cur = bare_header.find_next_sibling()
    grade_set = {"poor","fair","good","very good","excellent"}
    while cur and cur.name and not re.match(r"^h[1-6]$", cur.name, re.I):
        # capture pairs: model name and grade nearby
        txt = cur.get_text("\n", strip=True)
        if txt:
            # rough split by newlines; find lines that are grades and associate with preceding name
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            i = 0
            while i < len(lines)-1:
                name_line = lines[i]
                grade_line = lines[i+1].lower()
                if grade_line in grade_set:
                    # try to find details link near this block
                    a = cur.find("a", string=re.compile("View Details", re.I))
                    url = a.get("href") if a and a.has_attr("href") else None
                    if url and url.startswith("/"):
                        url = "https://ailuminate.mlcommons.org" + url
                    entries.append((name_line, lines[i+1], url))
                    i += 2
                else:
                    i += 1
        cur = cur.find_next_sibling()
    if not entries:
        # fallback regex
        text = soup.get_text("\n", strip=True)
        return _regex_extract_bare_models(text)
    return entries

def _regex_extract_bare_models(text: str) -> List[Tuple[str,str,Optional[str]]]:
    entries: List[Tuple[str,str,Optional[str]]] = []
    # Match lines like "Name ...\nVery Good" under "Bare Models" section
    parts = re.split(r"(?i)\bBare Models\b", text)
    chunk = parts[1] if len(parts) > 1 else text
    # capture pairs: a line then a grade on next line
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    grade_set = {"poor","fair","good","very good","excellent"}
    for i in range(len(lines)-1):
        if lines[i+1].lower() in grade_set:
            entries.append((lines[i], lines[i+1], None))
    # de-dup
    seen = set()
    deduped = []
    for n,g,u in entries:
        key = (n.lower(), g.lower())
        if key in seen: continue
        seen.add(key)
        deduped.append((n,g,u))
    return deduped

def find_ail_match(repo_tokens: List[str], ail_entries: List[Tuple[str,str,Optional[str]]]) -> List[Tuple[str,str,Optional[str]]]:
    """
    Conservative matching: the set of repo tokens must be a subset of tokens from the AIL name.
    """
    matches: List[Tuple[str,str,Optional[str]]] = []
    repo_set = set(repo_tokens)
    for name, grade, url in ail_entries:
        name_tokens = set(normalize_tokens(name))
        if repo_set and repo_set.issubset(name_tokens):
            matches.append((name, grade, url))
    return matches

def process_file(path: str, force: bool=False, add_comment: bool=False, dry_run: bool=False, timeout: int=30, write_path: Optional[str]=None) -> Tuple[bool, List[str], str, Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"[{path}] ERROR: cannot read JSON: {e}")
        return False, [], os.path.basename(path), write_path or path

    repo = hf_repo_from_aipackage(doc)
    if not repo:
        print(f"[{path}] not found: huggingface repo id in AIPackage")
        return False, [], os.path.basename(path), write_path or path

    tokens = core_repo_tokens(repo)
    if not tokens:
        print(f"[{path}] not found: normalized name tokens from repo '{repo}'")
        return False, [], repo, write_path or path

    try:
        entries = fetch_ailuminate_bare_models(timeout=timeout)
    except Exception as e:
        print(f"[{path}] ERROR: cannot fetch AILuminate page: {e}")
        return False, [], repo, write_path or path

    cand = find_ail_match(tokens, entries)
    if not cand:
        print(f"[{path}] not found: AILuminate v1.0 Bare Models match for repo '{repo}'")
        return False, [], repo, write_path or path
    if len(cand) > 1:
        names = "; ".join([c[0] for c in cand[:5]])
        print(f"[{path}] ambiguous: multiple AILuminate matches -> {names}")
        return False, [], repo, write_path or path

    name, grade, url = cand[0]
    risk = GRADE_TO_RISK.get(grade.lower())
    if not risk:
        print(f"[{path}] ERROR: unknown grade '{grade}' for '{name}'")
        return False, [], repo, write_path or path

    # locate AIPackage
    aip = None
    for elem in doc.get("element", []):
        if isinstance(elem, dict) and elem.get("type") == "AIPackage":
            aip = elem
            break
    if not aip:
        print(f"[{path}] ERROR: AIPackage element not found")
        return False, [], repo, write_path or path

    display_name = repo or os.path.basename(path)

    changed: List[str] = []
    already = aip.get("safetyRiskAssessment")
    if already and not force:
        pass
    else:
        aip["safetyRiskAssessment"] = risk
        changed.append("safetyRiskAssessment")

    if add_comment:
        prefix = aip.get("comment", "")
        note = f"[AILuminate v1.0] grade={grade} (source: {url or AIL_PAGE})"
        new_comment = (prefix + ("\n" if prefix else "") + note).strip()
        if new_comment != prefix:
            aip["comment"] = new_comment
            changed.append("comment")

    # Per-field reporting
    print(f"[{display_name}] found safetyRiskAssessment:")
    print(json.dumps(risk, ensure_ascii=False, indent=2))
    if add_comment:
        print(f"[{display_name}] found comment:")
        print(json.dumps(f"[AILuminate v1.0] grade={grade} (source: {url or AIL_PAGE})", ensure_ascii=False, indent=2))

    target_path = write_path or path
    is_inplace = (target_path == path)

    if dry_run:
        if changed:
            print(f"[{display_name}] would update fields: {', '.join(sorted(set(changed)))}")
        else:
            print(f"[{display_name}] no changes (dry-run)")
        return True, sorted(set(changed)), display_name, write_path or path

    must_write = True if (not is_inplace) else bool(changed)
    if not must_write:
        print(f"[{display_name}] no changes")
        return True, sorted(set(changed)), display_name, write_path or path

    try:
        parent_dir = os.path.dirname(os.path.abspath(target_path))
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        if is_inplace:
            print(f"[{display_name}] updated fields: {', '.join(sorted(set(changed)))}")
        else:
            if changed:
                print(f"[{display_name}] wrote enriched output to '{target_path}' (updated fields: {', '.join(sorted(set(changed)))})")
            else:
                print(f"[{display_name}] wrote output to '{target_path}' (no changes)")
    except Exception as e:
        print(f"[{path}] ERROR: cannot write JSON: {e}")
        return False, sorted(set(changed)), display_name, target_path
    return True, sorted(set(changed)), display_name, target_path

def main():
    ap = argparse.ArgumentParser(description="Enrich SPDX 3.0 AI-BOM with safetyRiskAssessment from MLCommons AILuminate v1.0 (Bare Models).")
    ap.add_argument("inputs", nargs="+", help="Input SPDX 3.0 AI-BOM JSON files.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing safetyRiskAssessment if present.")
    ap.add_argument("--add-comment", action="store_true", help="Append a brief AILuminate provenance note to AIPackage.comment.")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    # Default dry-run unless -o/--output is provided
    ap.add_argument("--dry-run", action="store_true", default=None, help="Do not write changes; only print decisions (default unless -o/--output is provided).")
    # Output behavior unified with HF enricher
    ap.add_argument("-o", "--output", nargs="?", const="__DEFAULT__", help="Output target. Use 'orig'/'inplace'/'overwrite' to modify input files in place; provide no value to write per-input as 'enriched.<basename>'; or provide a filename (only with a single input).")
    args = ap.parse_args()

    using_output = args.output is not None
    effective_dry_run = (args.dry_run if args.dry_run is not None else (not using_output))

    if using_output and args.output not in ("__DEFAULT__", None):
        val = str(args.output)
        key = val.lower()
        if key in {"orig","inplace","overwrite"}:
            pass
        else:
            if len(args.inputs) != 1:
                print("ERROR: Providing an explicit output filename with -o/--output requires exactly one input file.")
                sys.exit(2)

    ok = True
    summary_rows: List[Tuple[str, str, List[str]]] = []
    for p in args.inputs:
        write_path: Optional[str] = None
        if using_output:
            if args.output == "__DEFAULT__":
                base = os.path.basename(p)
                write_path = os.path.join(os.path.dirname(p), f"enriched.{base}")
            elif str(args.output).lower() in {"orig","inplace","overwrite"}:
                write_path = p
            else:
                write_path = str(args.output)

        succeeded, changed_fields, model_name, target_path = process_file(
            p,
            force=args.force,
            add_comment=args.add_comment,
            dry_run=effective_dry_run,
            timeout=args.timeout,
            write_path=write_path,
        )
        ok = succeeded and ok
        if using_output:
            summary_rows.append((model_name, target_path or (write_path or p), changed_fields))

    if using_output:
        printed_header = False
        for model_name, tgt, changed_fields in summary_rows:
            if not printed_header:
                print("=== Write Summary ===")
                printed_header = True
            if changed_fields:
                print(f"[{model_name}] wrote: {', '.join(changed_fields)} -> {tgt}")
            else:
                print(f"[{model_name}] wrote: no changes -> {tgt}")

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()


