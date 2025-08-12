
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hf2spdx-ai-bom.py (v1.3)
# SPDX 3.0 (JSON-LD) AI-BOM generator for Hugging Face models
# - Commit-pinned downloadLocation
# - All files listed via API siblings; contains relationships
# - SHA256 via LFS metadata -> blob "Raw pointer details" -> raw hashing for small files
# - License normalization: SPDX LicenseExpression when possible; otherwise SimpleLicensingText, preferring LICENSE file content
# - dataLicense uses LicenseExpression("CC0-1.0")
# - Dataset profile only when signal detected or --force-dataset; avoid non-vocabulary datasetAvailability; datasetType inferred from pipeline
# - typeOfModel/domain inferred from config.json + pipeline_tag (e.g., Decoder-only Transformer, Vision Transformer, CV/NLP)
# - Dependencies: transformers (with >= hint), optional onnxruntime/pillow/torch from README
#
# Usage:
#   python hf2spdx-ai-bom_v1_3.py <repo_id_or_url> -o out.json [--force-dataset] [--validate-minimal]

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
    "epl-2.0": "EPL-2.0",
    "mpl-2.0": "MPL-2.0",
    "agpl-3.0": "AGPL-3.0-only",
    "agpl-3.0-only": "AGPL-3.0-only",
    "agpl-3.0-or-later": "AGPL-3.0-or-later",
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
    "LICENSE",
    "LICENSE.txt",
    "README.md",
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
    custom_license_text_if_present: Long block of license text if page contains it.
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
    # Heuristic: capture community license blocks (e.g., LLAMA 3.1 COMM. LICENSE AGREEMENT)
    m = re.search(r"([A-Z][A-Z \d\.\-]*LICENSE AGREEMENT.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        block = m.group(1)
        custom_license_text = block[:20000]
    return lic, doi, custom_license_text

def find_pipeline_tag_from_api(model_info: Dict[str, Any]) -> Optional[str]:
    if isinstance(model_info, dict):
        if model_info.get("pipeline_tag"):
            return model_info["pipeline_tag"]
        tags = model_info.get("tags") or []
        for t in tags:
            if t in {"text-generation", "fill-mask", "token-classification", "image-classification",
                     "image-text-to-text", "image-to-text", "audio-classification"}:
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

def sha_from_lfs_meta(f: Dict[str, Any]) -> Optional[str]:
    lfs = f.get("lfs") or {}
    if not isinstance(lfs, dict):
        return None
    if "sha256" in lfs and isinstance(lfs["sha256"], str):
        return lfs["sha256"]
    oid = lfs.get("oid")
    if isinstance(oid, str):
        m = re.match(r"sha256:([0-9a-f]{64})$", oid)
        if m:
            return m.group(1)
    return None

def normalize_spdx_license(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    key = s.strip().lower()
    return SPDX_LICENSE_CANON.get(key, None)

def get_readme_text(model_info: Dict[str, Any], timeout: int = 30) -> str:
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
    try:
        r = requests.get(f"{HF_BASE}/{model_id}", timeout=timeout)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    except Exception:
        pass
    return ""

def parse_transformers_min_version(text: str) -> Optional[str]:
    # Detect "transformers >= 4.51.0" etc.
    m = re.search(r"transformers\s*(?:>=|=>)\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"pip\s+install.*transformers>=\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

def detect_extra_packages(text: str) -> List[Tuple[str, Optional[str]]]:
    pkgs = []
    if re.search(r"\bonnxruntime\b", text, re.IGNORECASE):
        pkgs.append(("onnxruntime", None))
    if re.search(r"\bfrom\s+PIL\b|\bpillow\b", text, re.IGNORECASE):
        pkgs.append(("pillow", None))
    if re.search(r"\btorch\b", text, re.IGNORECASE):
        pkgs.append(("torch", None))
    return pkgs

def try_read_config_json(repo_id: str, model_info: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
    commit_sha = model_info.get("sha") or "main"
    for f in model_info.get("siblings", []) or []:
        name = f.get("rfilename") or f.get("filename")
        if name == "config.json":
            url = f"{HF_BASE}/{repo_id}/resolve/{commit_sha}/{quote(name)}"
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200:
                    return json.loads(r.text)
            except Exception:
                return None
    return None

def infer_type_and_domain(pipeline_tag: Optional[str], cfg: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    # Domain
    dom = ["NLP"]
    if pipeline_tag:
        pt = pipeline_tag.lower()
        if "image" in pt: dom = ["CV"]
        elif "audio" in pt: dom = ["Audio"]
        elif "multimodal" in pt or "image-text" in pt: dom = ["Multimodal"]
        else: dom = ["NLP"]
    # Type of model
    t = "Transformer"
    key = ""
    if cfg:
        archs = cfg.get("architectures")
        mt = cfg.get("model_type")
        if isinstance(archs, list) and archs:
            key = str(archs[0]).lower()
        elif isinstance(mt, str):
            key = mt.lower()
    if "llama" in key: t = "Decoder-only Transformer"
    elif "vit" in key: t = "Vision Transformer"
    elif "clip" in key: t = "CLIP"
    elif "yolo" in key or "det" in key: t = "Detector"
    return t, dom

def infer_dataset_presence(readme_text: str) -> Tuple[bool, Optional[str]]:
    low = readme_text.lower()
    kws = [
        "dataset", "datasets", "webtext", "c4", "the pile", "pile", "common crawl",
        "pretrain", "fine-tune", "fine tune", "training data", "proprietary dataset",
        "imagenet-21k", "80,000 images", "80000 images"
    ]
    has = any(k in low for k in kws)
    label = None
    m = re.search(r"(\d{2,3}[, ]?\d{3})\s+images", low)
    if m:
        label = f"Approximately {m.group(1).replace(' ', '')} images"
    return has, label

def try_read_license_file(repo_id: str, model_info: Dict[str, Any], timeout: int = 30) -> Optional[str]:
    commit_sha = model_info.get("sha") or "main"
    candidates = [f.get("rfilename") or f.get("filename") for f in (model_info.get("siblings") or [])]
    for name in (candidates or []):
        if not name:
            continue
        low = name.lower()
        if low == "license" or low.startswith("license"):
            url = f"{HF_BASE}/{repo_id}/resolve/{commit_sha}/{quote(name)}"
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200 and r.text.strip():
                    return r.text
            except Exception:
                pass
    return None

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

    readme_text = get_readme_text(model_info)
    # Prefer cardData.datasets if present
    card = model_info.get("cardData") or {}
    datasets_meta = card.get("datasets") if isinstance(card, dict) else None
    ds_signal, ds_label_hint = infer_dataset_presence(readme_text)
    has_dataset_info = bool(datasets_meta) or ds_signal

    profile = ["core", "software", "simpleLicensing", "ai"]
    if has_dataset_info or force_dataset:
        profile.append("dataset")

    commit_sha = model_info.get("sha") or "main"

    # Try config.json to infer type/domain
    cfg = try_read_config_json(repo_id, model_info)
    t_model, dom = infer_type_and_domain(pipeline_tag, cfg)

    # Document
    doc = {
        "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
        "type": "SpdxDocument",
        "spdxId": "SPDXRef-DOCUMENT",
        "name": f"AI-BOM for {repo_id}",
        "profileConformance": profile,
        "dataLicense": {
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
                {"type": "Tool", "spdxId": "SPDXRef-TOOL", "name": "hf2spdx-ai-bom", "version": "1.3.0"}
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
        "typeOfModel": t_model,
        "domain": dom,
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

    # Dependencies
    xdeps: List[Tuple[str, Optional[str]]] = [("transformers", None)]
    min_tf = parse_transformers_min_version(readme_text)
    if min_tf:
        xdeps[0] = ("transformers", f"Requires transformers >= {min_tf} per model card")
    xdeps.extend(detect_extra_packages(readme_text))

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

    # Model Licensing (declared & concluded)
    lic_std = normalize_spdx_license(license_raw)
    if lic_std:
        for sid, rtype in (("SPDXRef-LIC-MODEL-DECLARED", "hasDeclaredLicense"),
                           ("SPDXRef-LIC-MODEL-CONCLUDED", "hasConcludedLicense")):
            doc["element"].append({"type": "LicenseExpression", "spdxId": sid, "licenseExpression": lic_std})
            rels.append({"type": "Relationship", "spdxId": f"SPDXRef-REL-{sid.split('-')[-1]}",
                         "from": "SPDXRef-MODEL", "to": sid, "relationshipType": rtype})
    else:
        # Prefer LICENSE file content if available
        # Note: custom_license_text will be filled by LICENSE file content in main() if available
        custom_text = custom_license_text or (license_raw or "Custom license (text unavailable)")
        for sid, rtype in (("SPDXRef-LIC-MODEL-DECLARED", "hasDeclaredLicense"),
                           ("SPDXRef-LIC-MODEL-CONCLUDED", "hasConcludedLicense")):
            doc["element"].append({"type": "SimpleLicensingText", "spdxId": sid, "licenseText": custom_text})
            rels.append({"type": "Relationship", "spdxId": f"SPDXRef-REL-{sid.split('-')[-1]}",
                         "from": "SPDXRef-MODEL", "to": sid, "relationshipType": rtype})

    # Dataset
    dataset_added = False
    if has_dataset_info or force_dataset:
        ds_name = None
        if datasets_meta:
            if isinstance(datasets_meta, list) and datasets_meta:
                ds_name = ", ".join(map(str, datasets_meta[:3])) + ("..." if len(datasets_meta) > 3 else "")
        if not ds_name:
            ds_name = ds_label_hint or ("See model card" if has_dataset_info else "Not disclosed")

        # datasetType from pipeline
        ds_type = "text"
        if pipeline_tag:
            pt = pipeline_tag.lower()
            if "image" in pt: ds_type = "image"
            elif "audio" in pt: ds_type = "audio"

        ds = {
            "type": "DatasetPackage",
            "spdxId": "SPDXRef-DATASET-TRAIN",
            "name": ds_name,
            "datasetType": ds_type,
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

    # Describes + relationships
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

    if validate_minimal and ("dataset" in doc["profileConformance"]) and (not dataset_added):
        ds_type = "text"
        if pipeline_tag:
            pt = pipeline_tag.lower()
            if "image" in pt: ds_type = "image"
            elif "audio" in pt: ds_type = "audio"
        ds = {
            "type": "DatasetPackage",
            "spdxId": "SPDXRef-DATASET-PLACEHOLDER",
            "name": "Not disclosed",
            "datasetType": ds_type,
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

    # Prefer LICENSE file content if non-SPDX
    lic_text_file = try_read_license_file(repo_id, model_info, timeout=args.timeout)
    if lic_text_file:
        lic_text = lic_text_file

    # 3) Pipeline tag
    pipeline_tag = find_pipeline_tag_from_api(model_info)

    # 4) Files + SHA256
    siblings = list_model_files_from_api(model_info)
    commit_sha = model_info.get("sha")
    sha_map: Dict[str, Optional[str]] = {}
    for f in siblings:
        fname = f["name"]
        sha_val = sha_from_lfs_meta(f)  # 1) LFS metadata
        try:
            if not sha_val and (fname.endswith((".safetensors", ".bin", ".onnx", ".pt", ".ckpt", ".index.json")) or (fname in COMMON_CONFIG_FILES)):
                sha_val = try_fetch_sha256_from_blob(repo_id, fname, commit_sha, timeout=args.timeout)  # 2) blob page
            if (not sha_val) and (f.get("size") is not None) and (f["size"] <= RAW_HASH_MAX):
                sha_val = try_compute_sha256_from_raw(repo_id, fname, commit_sha, max_bytes=RAW_HASH_MAX, timeout=args.timeout)  # 3) raw small files
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
