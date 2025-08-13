#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup

try:
    import yaml  # optional
except Exception:
    yaml = None

HF_BASE = "https://huggingface.co"
API_MODEL = HF_BASE + "/api/models/{repo_id}"

FIELDS = [
    "autonomyType",
    "domain",
    "informationAboutApplication",
    "informationAboutTraining",
    "limitation",
    "hyperparameter",
    "metric",
    "metricDecisionThreshold",
]

# ------------------ Helpers ------------------
def _detect_repo_from_doc(doc: Dict[str, Any]) -> Optional[str]:
    # prefer AIPackage.name (usually "owner/repo")
    model = None
    for elem in doc.get("element", []):
        if isinstance(elem, dict) and elem.get("type") == "AIPackage":
            model = elem
            break
    if not model:
        return None
    # name or downloadLocation
    name = model.get("name")
    if isinstance(name, str) and "/" in name:
        return name
    dl = model.get("downloadLocation") or ""
    if isinstance(dl, str):
        try:
            u = urlparse(dl)
            if "huggingface.co" in (u.netloc or ""):
                # /<owner>/<repo>/...
                parts = [p for p in u.path.strip("/").split("/") if p]
                if len(parts) >= 2:
                    return f"{parts[0]}/{parts[1]}"
        except Exception:
            pass
    return None

def _get_model_info(repo_id: str, timeout: int=30) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(API_MODEL.format(repo_id=repo_id), timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def _get_readme_text(repo_id: str, commit: Optional[str], timeout: int=30) -> str:
    # Try raw README from siblings with pinned commit, else fallback to page text
    try:
        info = _get_model_info(repo_id, timeout=timeout)
        sha = commit or (info.get("sha") if info else None) or "main"
        if info:
            for s in info.get("siblings", []) or []:
                name = s.get("rfilename") or s.get("filename")
                if name and name.lower() in {"readme.md","modelcard.md"}:
                    url = f"{HF_BASE}/{repo_id}/resolve/{sha}/{name}"
                    rr = requests.get(url, timeout=timeout)
                    if rr.status_code == 200 and rr.text:
                        return rr.text
    except Exception:
        pass
    # fallback: render page and strip text
    try:
        r = requests.get(f"{HF_BASE}/{repo_id}", timeout=timeout)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser").get_text("\n", strip=True)
    except Exception:
        pass
    return ""

def _safe_strip(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def _as_list_str(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if isinstance(x, (str, int, float)) and str(x).strip()]
    if isinstance(value, (str, int, float)):
        s = str(value).strip()
        return [s] if s else []
    return []

def _is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) == 0
    return False

def _merge_autonomy_type(existing: Any, incoming: Any) -> Any:
    ex_list = _as_list_str(existing)
    in_list = _as_list_str(incoming)
    if not ex_list:
        return in_list[0] if len(in_list) == 1 else in_list
    # union, preserve order: existing then new
    seen: Set[str] = set()
    out: List[str] = []
    for s in ex_list + in_list:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[0] if len(out) == 1 else out

def _merge_domain(existing: Any, incoming: Any) -> List[str]:
    ex_list = _as_list_str(existing)
    in_list = _as_list_str(incoming)
    seen: Set[str] = set()
    out: List[str] = []
    for s in ex_list + in_list:
        if s not in seen and s:
            seen.add(s)
            out.append(s)
    return out

def _normalize_hparams_list(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        for k in value.keys():
            out.append({"name": str(k), "value": value[k]})
        return out
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("param") or item.get("key")
                if name is None:
                    continue
                out.append({"name": str(name), "value": item.get("value")})
    return out

def _merge_hyperparameters(existing: Any, incoming: Any) -> List[Dict[str, Any]]:
    ex_list = _normalize_hparams_list(existing)
    in_list = _normalize_hparams_list(incoming)
    # ordered merge keyed by 'name': update existing values, append new ones
    name_to_value: Dict[str, Any] = {}
    order: List[str] = []
    for item in ex_list:
        n = item.get("name")
        if not isinstance(n, str):
            continue
        if n not in name_to_value:
            order.append(n)
        name_to_value[n] = item.get("value")
    for item in in_list:
        n = item.get("name")
        if not isinstance(n, str):
            continue
        if n not in name_to_value:
            order.append(n)
        # overwrite with new value when incoming provides it
        name_to_value[n] = item.get("value")
    return [{"name": n, "value": name_to_value.get(n)} for n in order]

def _metric_signature(m: Dict[str, Any]) -> Tuple:
    return (
        m.get("type"), m.get("name"), m.get("value"), m.get("unit"),
        m.get("dataset"), m.get("split"), m.get("config")
    )

def _normalize_metrics_list(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                rec: Dict[str, Any] = {}
                for k in ("type","name","value","unit","dataset","split","config"):
                    if k in item:
                        rec[k] = item[k]
                if rec:
                    out.append(rec)
    return out

def _merge_metrics(existing: Any, incoming: Any) -> List[Dict[str, Any]]:
    ex_list = _normalize_metrics_list(existing)
    in_list = _normalize_metrics_list(incoming)
    seen = set(_metric_signature(m) for m in ex_list)
    out = list(ex_list)
    for m in in_list:
        sig = _metric_signature(m)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(m)
    return out

def _merge_scalar(existing: Any, incoming: Any) -> Any:
    # For string-like fields: keep existing if non-empty; otherwise use incoming
    if _is_empty_value(existing):
        return incoming
    return existing

def _merge_threshold(existing: Any, incoming: Any) -> Any:
    if existing is None:
        return incoming
    return existing

def _compute_final_values(model: Dict[str, Any], add: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    final: Dict[str, Any] = {}
    changed: List[str] = []
    for k in FIELDS:
        incoming = add.get(k)
        existing = model.get(k)
        if k == "autonomyType":
            result = _merge_autonomy_type(existing, incoming)
        elif k == "domain":
            result = _merge_domain(existing, incoming)
        elif k == "hyperparameter":
            result = _merge_hyperparameters(existing, incoming)
        elif k == "metric":
            result = _merge_metrics(existing, incoming)
        elif k in ("informationAboutApplication","informationAboutTraining","limitation"):
            result = _merge_scalar(existing, incoming)
        elif k == "metricDecisionThreshold":
            result = _merge_threshold(existing, incoming)
        else:
            result = incoming if not _is_empty_value(incoming) else existing

        # Only include in final if either existing or incoming is non-empty
        if not _is_empty_value(result):
            final[k] = result
            # Determine changed if different from existing
            if json.dumps(existing, sort_keys=True, ensure_ascii=False) != json.dumps(result, sort_keys=True, ensure_ascii=False):
                changed.append(k)
    return final, changed

# ------------------ Extractors ------------------
def extract_from_carddata(card: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # 1) autonomyType / domain (only if explicitly present)
    if isinstance(card.get("autonomyType"), (str, list)):
        out["autonomyType"] = card["autonomyType"]
    if "domain" in card:
        dom = card.get("domain")
        if isinstance(dom, str):
            out["domain"] = [dom]
        elif isinstance(dom, list):
            out["domain"] = [d for d in dom if isinstance(d, str)]

    # 2) informationAboutApplication / informationAboutTraining / limitation
    for k_card, k_out in [
        ("informationAboutApplication", "informationAboutApplication"),
        ("informationAboutTraining", "informationAboutTraining"),
        ("limitations", "limitation"),  # some cards use plural
        ("limitation", "limitation"),
    ]:
        v = card.get(k_card)
        if isinstance(v, str) and _safe_strip(v):
            out[k_out] = _safe_strip(v)

    # 3) hyperparameters (common shapes: dict or list of kv pairs)
    for key in ("hyperparameters","training_hyperparameters","training-hyperparameters"):
        hp = card.get(key)
        if isinstance(hp, dict):
            hp_list = [{"name": k, "value": hp[k]} for k in sorted(hp.keys())]
            if hp_list:
                out["hyperparameter"] = hp_list
                break
        elif isinstance(hp, list):
            normalized = []
            for item in hp:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("param") or item.get("key")
                    if name is not None:
                        normalized.append({"name": str(name), "value": item.get("value")})
            if normalized:
                out["hyperparameter"] = normalized
                break

    # 4) metrics from model-index
    metrics = []
    mi = card.get("model-index")
    if isinstance(mi, list):
        for entry in mi:
            results = entry.get("results") if isinstance(entry, dict) else None
            if not isinstance(results, list): continue
            for r in results:
                mlist = r.get("metrics") if isinstance(r, dict) else None
                if not isinstance(mlist, list): continue
                for m in mlist:
                    if not isinstance(m, dict): continue
                    rec = {}
                    for k in ("type","name","value","unit","dataset","split","config"):
                        if k in m:
                            rec[k] = m[k]
                    if rec:
                        metrics.append(rec)
    if metrics:
        out["metric"] = metrics

    # 5) metricDecisionThreshold (if explicitly present in card)
    for k in ("metricDecisionThreshold","decision_threshold","threshold"):
        v = card.get(k)
        if isinstance(v, (int, float)):
            out["metricDecisionThreshold"] = v
            break
        if isinstance(v, dict):
            out["metricDecisionThreshold"] = v
            break

    return out

def extract_from_readme(readme_md: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    text = readme_md

    # Sections by headings
    def section_after(patterns: List[str], max_chars=4000) -> Optional[str]:
        # find the first matching heading and take until next heading of same/larger level
        lines = text.splitlines()
        idxs = []
        pat_re = re.compile("|".join(patterns), re.IGNORECASE)
        for i, line in enumerate(lines):
            if pat_re.search(line):
                idxs.append(i)
                break
        if not idxs:
            return None
        start = idxs[0] + 1
        # find next heading
        end = len(lines)
        for j in range(start, len(lines)):
            if re.match(r"^\s{0,3}#{1,6}\s+\S", lines[j]):
                end = j
                break
        chunk = "\n".join(lines[start:end])
        chunk = re.sub(r"\n{3,}", "\n\n", chunk).strip()
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars] + " …"
        return _safe_strip(chunk)

    # informationAboutApplication
    app = section_after([r"^#{1,6}\s*(Intended uses|Use cases|Applications?)\b"])
    if app: out["informationAboutApplication"] = app

    # informationAboutTraining
    train = section_after([r"^#{1,6}\s*(Training (data|procedure)|Fine-?tuning details|Pretraining data)\b"])
    if train: out["informationAboutTraining"] = train

    # limitation
    lim = section_after([r"^#{1,6}\s*(Limitations?|Known issues)\b"])
    if lim: out["limitation"] = lim

    # hyperparameters from code blocks or lists with key: value
    hp = []
    # YAML/INI-style lines
    for m in re.finditer(r"(?mi)^(?:-?\s*)?([A-Za-z0-9_.-]{2,})\s*:\s*([^\n]+)$", text):
        key, val = m.group(1), m.group(2).strip()
        # avoid catching headings and URLs
        if key.lower() in {"http","https","license","limitations","dataset","datasets","metric","metrics"}:
            continue
        if re.match(r"^https?://", val):
            continue
        hp.append({"name": key, "value": val})
        if len(hp) >= 50:
            break
    if hp:
        out["hyperparameter"] = hp

    # metrics from simple tables like | metric | value |
    metrics = []
    for tbl in re.finditer(r"(?mis)^\s*\|([^|]+)\|\s*([^\n|]+)\|\s*$", text):
        # very naive; we avoid parsing README-wide tables – only simple "Metric | Value" lines
        name = tbl.group(1).strip()
        value = tbl.group(2).strip()
        if name and re.match(r"^[A-Za-z0-9_. -]{2,}$", name):
            try:
                num = float(re.sub(r"[^0-9eE.+-]", "", value))
            except Exception:
                continue
            metrics.append({"name": name, "value": num})
            if len(metrics) >= 50:
                break
    if metrics and "metric" not in out:
        out["metric"] = metrics

    # metricDecisionThreshold textual patterns
    th = None
    m = re.search(r"(?i)\b(decision\s*)?threshold\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            th = float(m.group(2))
        except Exception:
            th = None
    if th is not None:
        out["metricDecisionThreshold"] = th

    # autonomyType/domain only if explicit words appear
    m = re.search(r"(?i)\bautonomy\s*type\s*[:=]\s*([A-Za-z-]+)", text)
    if m:
        out["autonomyType"] = m.group(1).strip()
    # domain as a list from lines like "Domain: healthcare, radiology"
    m = re.search(r"(?i)\bdomain\s*[:=]\s*([A-Za-z0-9,; \-/]+)", text)
    if m:
        doms = [d.strip() for d in re.split(r"[;,/]", m.group(1)) if d.strip()]
        if doms:
            out["domain"] = doms

    return out

def merge_into_aipackage(doc: Dict[str, Any], add: Dict[str, Any], verbose: bool=False) -> Tuple[Dict[str, Any], List[str]]:
    if not add:
        return {}, []
    # find AIPackage
    model = None
    for elem in doc.get("element", []):
        if isinstance(elem, dict) and elem.get("type") == "AIPackage":
            model = elem
            break
    if not model:
        if verbose: print("WARN: AIPackage not found in document")
        return {}, []
    final_values, changed = _compute_final_values(model, add)
    # write back merged values into model
    for k, v in final_values.items():
        model[k] = v
    return final_values, changed

# ------------------ Main ------------------
def process_file(path: str, timeout: int=30, dry_run: bool=False, verbose: bool=True, write_path: Optional[str]=None) -> Tuple[bool, List[str], str, Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"[{path}] ERROR: cannot read JSON: {e}")
        return False, [], os.path.basename(path), write_path or path

    repo_id = _detect_repo_from_doc(doc)
    if not repo_id:
        print(f"[{path}] ERROR: cannot detect Hugging Face repo id from document")
        return False, [], os.path.basename(path), write_path or path

    # for commit pin, try AIPackage.packageVersion if it looks like a SHA
    commit = None
    try:
        for elem in doc.get("element", []):
            if isinstance(elem, dict) and elem.get("type") == "AIPackage":
                pv = elem.get("packageVersion")
                if isinstance(pv, str) and re.fullmatch(r"[0-9a-f]{7,40}", pv):
                    commit = pv
                    break
    except Exception:
        pass

    info = _get_model_info(repo_id, timeout=timeout) or {}
    card = info.get("cardData") or {}
    add_card = extract_from_carddata(card)
    readme = _get_readme_text(repo_id, commit, timeout=timeout)
    add_readme = extract_from_readme(readme) if readme else {}

    # Merge preference: cardData > README
    merged = dict(add_readme)
    merged.update(add_card)

    if verbose:
        display_name = repo_id or os.path.basename(path)
        # Report not found per field
        for k in FIELDS:
            if k not in merged:
                print(f"[{display_name}] not found: {k}")
        # Report found fields as separate blocks per field
        found_keys = [k for k in FIELDS if k in merged]
        for k in found_keys:
            print(f"[{display_name}] found {k}:")
            try:
                print(json.dumps(merged[k], ensure_ascii=False, indent=2))
            except Exception:
                print(json.dumps(str(merged[k]), ensure_ascii=False, indent=2))

    final_values, changed = merge_into_aipackage(doc, merged, verbose=verbose)

    # Decide target output behavior
    target_path = write_path or path
    is_inplace = (target_path == path)

    if dry_run:
        if changed:
            # already printed the block above; keep summary concise
            if verbose:
                print(f"[{display_name}] would update fields: {', '.join(changed)}")
        else:
            if verbose: print(f"[{display_name}] no changes (dry-run)")
        return True, changed, display_name, write_path or path

    # Not dry-run: write output. If writing to a new file (not in-place), always write.
    must_write = True if (not is_inplace) else bool(changed)
    if not must_write:
        if verbose: print(f"[{display_name}] no changes")
        return True, changed, display_name, write_path or path

    try:
        # Ensure parent dir exists for non-inplace writes
        parent_dir = os.path.dirname(os.path.abspath(target_path))
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        if verbose:
            if is_inplace:
                print(f"[{display_name}] updated fields: {', '.join(changed)}")
            else:
                if changed:
                    print(f"[{display_name}] wrote enriched output to '{target_path}' (updated fields: {', '.join(changed)})")
                else:
                    print(f"[{display_name}] wrote output to '{target_path}' (no changes)")
    except Exception as e:
        print(f"[{path}] ERROR: cannot write JSON to '{target_path}': {e}")
        return False, changed, display_name, target_path

    return True, changed, display_name, target_path

def main():
    ap = argparse.ArgumentParser(description="Enrich SPDX 3.0 AI-BOM with optional AI fields from Hugging Face (facts only).")
    ap.add_argument("inputs", nargs="+", help="Input SPDX 3.0 AI-BOM JSON files.")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    # If --dry-run is not provided, default to True unless -o/--output is used
    ap.add_argument("--dry-run", action="store_true", default=None, help="Do not write changes; only print findings (default unless -o/--output is provided).")
    ap.add_argument("--quiet", action="store_true", help="Less verbose output")
    # -o/--output behavior:
    #   -o orig|inplace|overwrite  -> overwrite each input file in place
    #   -o (no value)              -> per-input output 'enriched.<basename>' alongside input
    #   -o <filename>              -> write to <filename> (only valid with single input)
    ap.add_argument("-o", "--output", nargs="?", const="__DEFAULT__", help="Output target. Use 'orig'/'inplace'/'overwrite' to modify input files in place; provide no value to write per-input as 'enriched.<basename>'; or provide a filename (only with a single input).")
    args = ap.parse_args()

    # Determine effective dry_run
    using_output = args.output is not None
    effective_dry_run = (args.dry_run if args.dry_run is not None else (not using_output))

    # Validate -o usage
    if using_output and args.output not in ("__DEFAULT__", None):
        val = str(args.output)
        key = val.lower()
        if key in {"orig","inplace","overwrite"}:
            pass
        else:
            # A concrete filename: only allowed when a single input
            if len(args.inputs) != 1:
                print("ERROR: Providing an explicit output filename with -o/--output requires exactly one input file.")
                sys.exit(2)

    ok = True
    summary_rows: List[Tuple[str, str, List[str]]] = []  # (model, target_path, changed_fields)
    for p in args.inputs:
        write_path: Optional[str] = None
        if using_output:
            if args.output == "__DEFAULT__":
                # per-input: enriched.<basename>
                base = os.path.basename(p)
                write_path = os.path.join(os.path.dirname(p), f"enriched.{base}")
            elif str(args.output).lower() in {"orig","inplace","overwrite"}:
                write_path = p  # in place
            else:
                write_path = str(args.output)  # explicit filename (single input validated above)

        succeeded, changed_fields, model_name, target_path = process_file(
            p,
            timeout=args.timeout,
            dry_run=effective_dry_run,
            verbose=not args.quiet,
            write_path=write_path,
        )
        ok = succeeded and ok
        if using_output:
            summary_rows.append((model_name, target_path or (write_path or p), changed_fields))

    # Final summary when -o/--output is used
    if using_output:
        # Group by target to avoid noise in single-file run
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


