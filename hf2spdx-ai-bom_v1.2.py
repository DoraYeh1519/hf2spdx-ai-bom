#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hf2spdx-ai-bom.py (v1.2)
# SPDX 3.0 (JSON-LD) AI-BOM generator for Hugging Face models
# Uses HF API + lightweight scraping
# Improvements in v1.2:
#   1) Pin file downloadLocation to commit SHA (not "main") for reproducibility
#   2) For each file: create File + contains; fileKind fixed to "file" per SPDX 3.0
#   3) Try to get SHA256 from "Raw pointer details"; else compute for small raw files (<5 MB)
#   4) Normalize model license: SPDX if known; else create SimpleLicensingText with scraped text (custom license)
#   5) dataLicense now uses LicenseExpression("CC0-1.0") instead of SimpleLicensingText
#   6) Dataset profile only when signal detected or --force-dataset; remove unsupported datasetAvailability values
#   7) Parse README to infer extra runtime deps (e.g., onnxruntime) and transformers >= version; attach via sourceInfo

import argparse
import hashlib
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote

import requests
from bs4 import BeautifulSoup

HF_BASE = "https://huggingface.co"
API_MODEL = HF_BASE + "/api/models/{repo_id}"

# SPDX License ID normalization (extend as needed)
SPDX_LICENSE_CANON = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache2": "Apache-2.0",
    "apache2.0": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "gpl-3.0": "GPL-3.0-only",
    "gpl-3.0-only": "GPL-3.0-only",
    "gpl-3.0-or-later": "GPL-3.0-or-later",
    "lgpl-3.0": "LGPL-3.0-only",
    "lgpl-3.0-only": "LGPL-3.0-only",
    "lgpl-3.0-or-later": "LGPL-3.0-or-later",
    "cc-by-4.0": "CC-BY-4.0",
    "cc0-1.0": "CC0-1.0",
    "bsd-2": "BSD-2-Clause",
    "bsd-3": "BSD-3-Clause",
}

RAW_HASH_MAX = 5 * 1024 * 1024  # 5 MB

COMMON_CONFIG_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "model.safetensors.index.json",
}

# ---------------- Helpers ----------------
def extract_repo_id_from_url_or_id(input_str: str) -> str:
    if input_str.startswith(("http://", "https://")):
        parsed = urlparse(input_str)
        if "huggingface.co" in parsed.netloc:
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            raise ValueError(f"Invalid Hugging Face URL: {input_str}")
        raise ValueError(f"Not a Hugging Face URL: {input_str}")
    return input_str

def get_model_info(repo_id: str, timeout: int = 30) -> Dict[str, Any]:
    url = API_MODEL.format(repo_id=repo_id)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_model_page(repo_id: str, timeout: int = 30) -> BeautifulSoup:
    r = requests.get(f"{HF_BASE}/{repo_id}", timeout=timeout)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def parse_license_and_doi_from_page(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns: (license_string, doi, custom_license_text_if_present)
    custom_license_text_if_present: Long block of license text if page contains it (e.g., Llama Community License).
    """
    text = soup.get_text("\n", strip=True)
    lic = None
    m = re.search(r"License:\s*([A-Za-z0-9.\-+]+)", text, re.IGNORECASE)
    if m:
        lic = m.group(1)

    doi = None
    m = re.search(r"\bdoi:\s*([0-9.]+/[A-Za-z0-9._/-]+)", text, re.IGNORECASE)
    if m:
        doi = m.group(1)

    custom_license_text = None
    # Heuristic: capture a large block if "COMMUNITY LICENSE" or "LICENSE AGREEMENT" appears
    m = re.search(r"(LLAMA.*?COMMUNITY LICENSE AGREEMENT.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        # Trim to a reasonable size to avoid megabytes; but keep full paragraphs
        block = m.group(1)
        custom_license_text = block[:20000]  # cap at ~20k chars
    return lic, doi, custom_license_text

def find_pipeline_tag_from_api(model_info: Dict[str, Any]) -> Optional[str]:
    if isinstance(model_info, dict):
        if model_info.get("pipeline_tag"):
            return model_info["pipeline_tag"]
        tags = model_info.get("tags") or []
        for t in tags:
            if t in {"text-generation", "fill-mask", "token-classification", "image-classification"}:
                return t
    return None

def list_model_files_from_api(model_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = []
    for f in model_info.get("siblings", []) or []:
        name = f.get("rfilename") or f.get("filename")
        if not name:
            continue
        entry = {"name": name}
        if "size" in f and isinstance(f["size"], int):
            entry["size"] = f["size"]
        if "lfs" in f and isinstance(f["lfs"], dict):
            entry["lfs"] = f["lfs"]
        files.append(entry)
    return files

def try_fetch_sha256_from_blob(repo_id: str, filename: str, commit_sha: Optional[str], timeout: int = 30) -> Optional[str]:
    """
    Prefer blob/{commit}/... to avoid "main" drift.
    """
    ref = commit_sha or "main"
    url = f"{HF_BASE}/{repo_id}/blob/{quote(ref)}/{quote(filename)}"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    m = re.search(r"SHA256:\s*([0-9a-f]{64})", text, re.IGNORECASE)
    return m.group(1) if m else None

def try_compute_sha256_from_raw(repo_id: str, filename: str, commit_sha: Optional[str], max_bytes: int = RAW_HASH_MAX, timeout: int = 30) -> Optional[str]:
    """
    Download raw and compute SHA256 for small files.
    """
    ref = commit_sha or "main"
    url = f"{HF_BASE}/{repo_id}/resolve/{quote(ref)}/{quote(filename)}"
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        h = hashlib.sha256()
        total = 0
        for chunk in r.iter_content(16384):
            if chunk:
                total += len(chunk)
                if total > max_bytes:
                    return None
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def normalize_spdx_license(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    key = s.strip().lower()
    return SPDX_LICENSE_CANON.get(key, None)

def infer_dataset_presence(readme_text: str) -> Tuple[bool, Optional[str]]:
    """
    Return (has_dataset_info, dataset_label)
    dataset_label is a short label we may use as DatasetPackage.name if we can infer something useful.
    """
    low = readme_text.lower()
    kws = [
        "dataset", "datasets", "webtext", "c4", "the pile", "pile", "common crawl",
        "pretrain", "fine-tune", "fine tune", "training data", "proprietary dataset", "imagenet-21k"
    ]
    has = any(k in low for k in kws)
    label = None
    m = re.search(r"(\d{2,3}[, ]?\d{3})\s+images", low)
    if m:
        label = f"Approximately {m.group(1).replace(' ', '')} images"
    return has, label

def get_readme_text(model_info: Dict[str, Any], timeout: int = 30) -> str:
    # Try to fetch README raw if listed
    model_id = model_info.get("modelId")
    for f in model_info.get("siblings", []) or []:
        name = f.get("rfilename") or f.get("filename")
        if name and name.lower() in {"readme.md", "modelcard.md"}:
            raw = f"{HF_BASE}/{model_id}/resolve/{model_info.get('sha','main')}/{quote(name)}"
            try:
                rr = requests.get(raw, timeout=timeout)
                if rr.status_code == 200:
                    return rr.text
            except Exception:
                pass
    # Fallback: page text
    try:
        soup = get_model_page(model_id, timeout=timeout)
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""

def parse_transformers_min_version(text: str) -> Optional[str]:
    # Look for patterns like "transformers >= 4.51.0" or "pip install -U transformers>=4.51.0"
    m = re.search(r"transformers\s*[>=!~]*\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

def detect_extra_packages(text: str) -> List[Tuple[str, Optional[str]]]:
    pkgs = []
    if re.search(r"\bonnxruntime\b", text, re.IGNORECASE):
        pkgs.append(("onnxruntime", None))
    if re.search(r"\bpillow\b|\bfrom\s+PIL\b", text, re.IGNORECASE):
        pkgs.append(("pillow", None))
    if re.search(r"\btorch\b", text, re.IGNORECASE):
        pkgs.append(("torch", None))
    return pkgs

# -------------- SPDX assembly --------------
def build_spdx(repo_id: str,
               model_info: Dict[str, Any],
               license_raw: Optional[str],
               custom_license_text: Optional[str],
               doi: Optional[str],
               files: List[Dict[str, Any]],
               sha_map: Dict[str, Optional[str]],
               pipeline_tag: Optional[str],
               force_dataset: bool,
               validate_minimal: bool) -> Dict[str, Any]:

    # Profiles (add dataset only if we have signal or forced)
    readme_text = get_readme_text(model_info)
    has_dataset_info, dataset_label = infer_dataset_presence(readme_text)
    profile = ["core", "software", "simpleLicensing", "ai"]
    if has_dataset_info or force_dataset:
        profile.append("dataset")

    commit_sha = model_info.get("sha") or "main"

    # Document
    doc = {
        "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
        "type": "SpdxDocument",
        "spdxId": "SPDXRef-DOCUMENT",
        "name": f"AI-BOM for {repo_id}",
        "profileConformance": profile,
        "dataLicense": {  # Use LicenseExpression for CC0-1.0
            "type": "LicenseExpression",
            "spdxId": "SPDXRef-DATA-LICENSE",
            "licenseExpression": "CC0-1.0"
        },
        "creationInfo": {
            "type": "CreationInfo",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "specVersion": "3.0.1",
            "createdBy": [
                {"type": "Organization", "spdxId": "SPDXRef-ORG", "name": "Hugging Face Hub Scraper (example)"}
            ],
            "createdUsing": [
                {"type": "Tool", "spdxId": "SPDXRef-TOOL", "name": "hf2spdx-ai-bom", "version": "1.2.0"}
            ]
        },
        "element": [],
        "rootElement": ["SPDXRef-MODEL"],
        "externalRef": []
    }

    # AIPackage
    desc = f"Model from {repo_id}"
    if pipeline_tag:
        desc = f"{pipeline_tag} model from {repo_id}"

    ai_pkg = {
        "type": "AIPackage",
        "spdxId": "SPDXRef-MODEL",
        "name": repo_id,
        "description": desc,
        "packageVersion": commit_sha,
        "downloadLocation": f"{HF_BASE}/{repo_id}",
        "standardCompliance": [],
        "typeOfModel": "Transformer",
        "domain": ["NLP"],
        "hyperparameter": {},
        "modelExplainability": None,
        "informationAboutTraining": "See model card" if has_dataset_info else "Not specified",
        "modelDataPreprocessing": "Not specified",
        "metric": [],
        "safetyRiskAssessment": [],
        "useSensitivePersonalInformation": False,
        "energyConsumption": []
    }
    doc["element"].append(ai_pkg)

    rels: List[Dict[str, Any]] = []

    # Files
    for idx, f in enumerate(sorted(files, key=lambda x: x["name"])):
        fname = f["name"]
        spdx_id = f"SPDXRef-FILE-{idx:04d}"
        hashes = []
        sha = sha_map.get(fname)
        if sha:
            hashes.append({"type": "Hash", "algorithm": "SHA256", "hashValue": sha})

        doc["element"].append({
            "type": "File",
            "spdxId": spdx_id,
            "name": fname,
            "fileKind": "file",
            "hash": hashes,
            "downloadLocation": f"{HF_BASE}/{repo_id}/resolve/{quote(commit_sha)}/{quote(fname)}"
        })
        rels.append({
            "type": "Relationship",
            "spdxId": f"SPDXRef-REL-CONTAINS-{idx:04d}",
            "from": "SPDXRef-MODEL",
            "to": spdx_id,
            "relationshipType": "contains"
        })

    # Dependencies from README
    # Always include transformers; add note if min version hinted
    xdeps: List[Tuple[str, Optional[str]]] = [("transformers", None)]
    min_tf = parse_transformers_min_version(readme_text)
    if min_tf:
        xdeps[0] = ("transformers", f"Requires transformers >= {min_tf} per model card")
    # Extra
    xdeps.extend(detect_extra_packages(readme_text))
    # Deduplicate by name
    seen = set()
    for name, note in xdeps:
        if name in seen:
            continue
        seen.add(name)
        pid = f"SPDXRef-PKG-{name.upper()}"
        pkg = {
            "type": "Package",
            "spdxId": pid,
            "name": name,
            "packageVersion": None,
            "downloadLocation": f"https://pypi.org/project/{name}/",
            "packageUrl": f"pkg:pypi/{name}"
        }
        if note:
            pkg["sourceInfo"] = note
        doc["element"].append(pkg)
        rels.append({
            "type": "Relationship",
            "spdxId": f"SPDXRef-REL-DEP-{name.upper()}",
            "from": "SPDXRef-MODEL",
            "to": pid,
            "relationshipType": "dependsOn"
        })

    # Licensing for the model (declared/concluded)
    lic_std = normalize_spdx_license(license_raw)
    if lic_std:
        # Use LicenseExpression for standard SPDX id
        for sid, rtype in (("SPDXRef-LIC-MODEL-DECLARED", "hasDeclaredLicense"),
                           ("SPDXRef-LIC-MODEL-CONCLUDED", "hasConcludedLicense")):
            doc["element"].append({"type": "LicenseExpression", "spdxId": sid, "licenseExpression": lic_std})
            rels.append({"type": "Relationship", "spdxId": f"SPDXRef-REL-{sid.split('-')[-1]}",
                         "from": "SPDXRef-MODEL", "to": sid, "relationshipType": rtype})
    else:
        # Custom/non-SPDX license → SimpleLicensingText with scraped text if available
        custom_text = custom_license_text or (license_raw or "Custom license (text unavailable)")
        for sid, rtype in (("SPDXRef-LIC-MODEL-DECLARED", "hasDeclaredLicense"),
                           ("SPDXRef-LIC-MODEL-CONCLUDED", "hasConcludedLicense")):
            doc["element"].append({"type": "SimpleLicensingText", "spdxId": sid, "licenseText": custom_text})
            rels.append({"type": "Relationship", "spdxId": f"SPDXRef-REL-{sid.split('-')[-1]}",
                         "from": "SPDXRef-MODEL", "to": sid, "relationshipType": rtype})

    # Dataset (only when present/forced). No datasetAvailability unless we know a valid value.
    dataset_added = False
    if has_dataset_info or force_dataset:
        ds_name = dataset_label or ("See model card" if has_dataset_info else "Not disclosed")
        ds = {
            "type": "DatasetPackage",
            "spdxId": "SPDXRef-DATASET-TRAIN",
            "name": ds_name,
            "datasetType": ("image" if (pipeline_tag and "image" in pipeline_tag) else ("audio" if (pipeline_tag and "audio" in pipeline_tag) else "text")),
            "intendedUse": "Pretraining"
        }
        doc["element"].append(ds)
        rels.append({
            "type": "Relationship",
            "spdxId": "SPDXRef-REL-MODEL-TRAINEDON",
            "from": "SPDXRef-MODEL",
            "to": "SPDXRef-DATASET-TRAIN",
            "relationshipType": "trainedOn"
        })
        dataset_added = True

    # describes + relationships
    doc["element"].append({
        "type": "Relationship",
        "spdxId": "SPDXRef-REL-DESCRIBES",
        "from": "SPDXRef-DOCUMENT",
        "to": "SPDXRef-MODEL",
        "relationshipType": "describes"
    })
    doc["element"].extend(rels)

    if doi:
        doc["externalRef"].append({"type": "ExternalRef", "externalRefType": "doi", "locator": doi})

    # Minimal validation: if dataset profile present but no dataset element, inject placeholder
    if validate_minimal and ("dataset" in doc["profileConformance"]) and (not dataset_added):
        ds = {
            "type": "DatasetPackage",
            "spdxId": "SPDXRef-DATASET-PLACEHOLDER",
            "name": "Not disclosed",
            "datasetType": ("image" if (pipeline_tag and "image" in pipeline_tag) else ("audio" if (pipeline_tag and "audio" in pipeline_tag) else "text")),
            "intendedUse": "Unspecified"
        }
        doc["element"].append(ds)
        doc["element"].append({
            "type": "Relationship",
            "spdxId": "SPDXRef-REL-MODEL-TRAINEDON-PLACEHOLDER",
            "from": "SPDXRef-MODEL",
            "to": "SPDXRef-DATASET-PLACEHOLDER",
            "relationshipType": "trainedOn"
        })

    return doc

# -------------- Pipeline --------------
def main():
    ap = argparse.ArgumentParser(description="Generate SPDX 3.0 AI-BOM for a Hugging Face model (commit-pinned, license-normalized).")
    ap.add_argument("input", help="Hugging Face repo_id or URL, e.g., meta-llama/Llama-3.1-8B-Instruct or https://huggingface.co/Falconsai/nsfw_image_detection")
    ap.add_argument("-o", "--output", nargs="?", const="", help="Output file (default: <repo_id>.spdx3.json). If omitted, print to stdout.")
    ap.add_argument("--force-dataset", action="store_true", help="Force add placeholder DatasetPackage + trainedOn even if no dataset info detected")
    ap.add_argument("--validate-minimal", action="store_true", help="If dataset profile is present but no dataset element, auto-add placeholder")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    args = ap.parse_args()

    try:
        repo_id = extract_repo_id_from_url_or_id(args.input)
    except ValueError as e:
        print(f"[ERR] {e}", file=sys.stderr)
        sys.exit(1)

    # 1) API
    try:
        model_info = get_model_info(repo_id, timeout=args.timeout)
    except Exception as e:
        print(f"[ERR] Failed to read HF API: {e}", file=sys.stderr)
        sys.exit(2)

    # 2) License / DOI + possible custom license text
    lic_raw = doi = lic_text = None
    try:
        soup = get_model_page(repo_id, timeout=args.timeout)
        lic_raw, doi, lic_text = parse_license_and_doi_from_page(soup)
    except Exception:
        pass
    if not lic_raw:
        lic_raw = model_info.get("license")

    # 3) Pipeline tag
    pipeline_tag = find_pipeline_tag_from_api(model_info)

    # 4) Files + SHA256
    siblings = list_model_files_from_api(model_info)
    commit_sha = model_info.get("sha")
    sha_map: Dict[str, Optional[str]] = {}
    for f in siblings:
        fname = f["name"]
        sha_val = None
        try:
            # Try blob pointer SHA first
            if fname.endswith((".safetensors", ".bin", ".onnx", ".pt", ".ckpt", ".index.json")) or (fname in COMMON_CONFIG_FILES):
                sha_val = try_fetch_sha256_from_blob(repo_id, fname, commit_sha, timeout=args.timeout)
            # If still missing and small file, compute from raw
            if (not sha_val) and (f.get("size") is not None) and (f["size"] <= RAW_HASH_MAX):
                sha_val = try_compute_sha256_from_raw(repo_id, fname, commit_sha, max_bytes=RAW_HASH_MAX, timeout=args.timeout)
        except Exception:
            sha_val = None
        sha_map[fname] = sha_val

    # 5) Assemble SPDX
    spdx = build_spdx(
        repo_id=repo_id,
        model_info=model_info,
        license_raw=lic_raw,
        custom_license_text=lic_text,
        doi=doi,
        files=siblings,
        sha_map=sha_map,
        pipeline_tag=pipeline_tag,
        force_dataset=args.force_dataset,
        validate_minimal=args.validate_minimal
    )

    # 6) Output
    if args.output is not None:
        out = args.output if args.output else f"{repo_id.replace('/', '_')}.spdx3.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(spdx, f, ensure_ascii=False, indent=2)
        print(f"[OK] SPDX AI-BOM saved to: {out}")
    else:
        json.dump(spdx, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
