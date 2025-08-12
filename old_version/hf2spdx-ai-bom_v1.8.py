#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hf2spdx-ai-bom.py (v1.8)
# Policy: ONLY include fields that can be directly fetched from HF API/page/files.
# - No model-type/domain inference (omit unless explicitly found somewhere structured — not supported currently)
# - Dataset only if HF API cardData.datasets exists (or --force-dataset). No README keyword heuristics by default.
# - Dataset fields kept minimal (name only). No datasetType/intendedUse unless user forces with a flag.
# - Dependencies only from explicit evidence:
#     1) requirements.txt / environment.yml / pip/conda files in repo
#     2) README explicit "pip install X" lines or "import X" statements in code blocks
# - Keep commit-pinned downloadLocation for files and package.
# - dataLicense kept as CC0-1.0 (SPDX document requirement; not from HF)
# - Licensing: SPDX LicenseExpression if API/page license is SPDX; otherwise SimpleLicensingText using LICENSE file content if present.
#
# v1.8 changes:
#   * FIX: ignore requirement/constraint files after -r/--requirement/-c/--constraint in installer lines.
#   * Keep v1.7 improvements; no behavior regressions expected.
#
# v1.7 changes:
#   * FIX: removed duplicate get_requirements_from_readme definition that regressed filtering (flags/git).
#   * NEW: fallback DOI/arXiv extraction from README markdown (also supports https://doi.org/...).
#   * Tweak: more robust "License:" detection from README if page parsing fails.
#   * Minor: keep previous behaviors, IDs, and schema.

import argparse
import hashlib
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse, quote

import requests
from bs4 import BeautifulSoup

HF_BASE = "https://huggingface.co"
API_MODEL = HF_BASE + "/api/models/{repo_id}"

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
SMALL_FILE_EXTS = {".json",".md",".txt",".py",".ini",".cfg",".yaml",".yml",".jinja",".jinja2",".vocab",".merges",".bpe"}
SMALL_FILE_NAMES = {
    "config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
    "merges.txt","vocab.json","vocab.txt","special_tokens_map.json","chat_template.json",
    "chat_template.jinja","preprocessor_config.json","model.safetensors.index.json",
    "LICENSE","LICENSE.txt","README.md","labels.json","requirements.txt","environment.yml","environment.yaml"
}
BINARY_EXTS = {".safetensors",".bin",".onnx",".pt",".ckpt"}
COMMON_CONFIG_FILES = set(SMALL_FILE_NAMES)

# Patterns
ARXIV_RE = re.compile(r"\barxiv\s*[:=]\s*([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.IGNORECASE)
DOI_RE = re.compile(r"""\b(?:doi\s*[:/]\s*|https?://doi\.org/)\s*([0-9.]+/[A-Za-z0-9._:/\-]+)""", re.IGNORECASE)

INSTALL_CMD_RE = re.compile(
    r"""(?mx)              # verbose, multiline
    ^[^\n]*\b(?:uv\s+pip|pip3?|python\s+-m\s+pip|mamba|conda)\s+install\b   # installer
    (?P<tail>[^\n]*)       # rest of the line
    """
)

INSTALL_SPLIT_RE = re.compile(r"\s+")

BAD_TOKENS = {
    "-U","--upgrade","--user","--pre","-q","-qq","-y","--yes","-n","--dry-run",
    "-i","--index-url","--extra-index-url","--find-links","-f","--no-cache-dir",
    "--use-pep517","--break-system-packages","--trusted-host","--retries","--timeout",
}

IMPORT_ALLOWLIST = {
    "transformers","torch","tensorflow","keras","flax","accelerate","diffusers",
    "onnxruntime","pillow","numpy","matplotlib","scipy","safetensors","bitsandbytes",
    "sentencepiece","tokenizers","timm","opencv-python","opencv","gradio","datasets",
    "peft","trl","vllm","mlx","kernels","pydantic"
}

def normalize_install_line_tail(tail: str) -> str:
    # Remove continuation backslashes and trailing comments
    t = tail.replace("\\\n", " ").replace("\\", " ")
    t = re.sub(r"\s+#.*$", "", t)
    return t


def extract_pkgs_from_install_tail(tail: str) -> Set[str]:
    pkgs: Set[str] = set()
    t = normalize_install_line_tail(tail)
    tokens = INSTALL_SPLIT_RE.split(t.strip())
    REQ_MARKERS = {"-r", "--requirement", "-c", "--constraint", "--requirements", "--constraints"}
    skip_next_if_req = False
    for i, tok in enumerate(tokens):
        if not tok:
            continue
        if skip_next_if_req:
            skip_next_if_req = False
            continue
        if tok in REQ_MARKERS:
            skip_next_if_req = True
            continue
        if tok in BAD_TOKENS:
            continue
        if tok.startswith("-"):
            continue
        low = tok.lower().strip().strip(",")
        if low in {"requirements.txt","requirements-dev.txt","constraints.txt","constraints-dev.txt"}:
            continue
        if low.endswith((".txt", ".in", ".cfg", ".yaml", ".yml")):
            continue
        if "://" in low or low.startswith("git+") or low.startswith("file:") or low.endswith((".whl",".zip",".tar.gz",".tgz")):
            continue
        low = re.sub(r"\[.*\]$", "", low)
        low = re.split(r"[<>=!~]", low)[0]
        if not re.match(r"^[a-z0-9][a-z0-9._-]*$", low):
            continue
        if low in {"git","https","http","pip","uv","python","mamba","conda"}:
            continue
        if low == "opencv":
            low = "opencv-python"
        pkgs.add(low)
    return pkgs

def extract_pkgs_from_readme(readme_text: str) -> Set[str]:
    """Parse README to discover dependencies, using explicit commands/imports only."""
    pkgs = set()
    # 1) Parse installer lines
    for m in INSTALL_CMD_RE.finditer(readme_text):
        tail = m.group("tail")
        pkgs |= extract_pkgs_from_install_tail(tail)
    # 2) Parse import lines (allowlist only) to avoid pulling random module names
    for m in re.finditer(r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))",
                         readme_text, flags=re.IGNORECASE|re.MULTILINE):
        mod = (m.group(1) or m.group(2) or "").split(".")[0].lower()
        if mod in IMPORT_ALLOWLIST:
            if mod == "opencv":
                mod = "opencv-python"
            pkgs.add(mod)
    return pkgs

REQ_FILES = {"requirements.txt", "requirements-dev.txt", "requirements.in",
             "environment.yml", "environment.yaml", "Pipfile", "pyproject.toml", "setup.cfg", "setup.py"}

# ---------------- Helpers ----------------
def extract_repo_id_from_url_or_id(input_str: str) -> str:
    if input_str.startswith(("http://","https://")):
        parsed = urlparse(input_str)
        if "huggingface.co" in parsed.netloc:
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            raise ValueError(f"Invalid Hugging Face URL: {input_str}")
        raise ValueError(f"Not a Hugging Face URL: {input_str}")
    return input_str

def get_model_info(repo_id: str, timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(API_MODEL.format(repo_id=repo_id), timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_model_page(repo_id: str, timeout: int = 30) -> BeautifulSoup:
    r = requests.get(f"{HF_BASE}/{repo_id}", timeout=timeout)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def parse_license_and_doi_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    lic = None
    m = re.search(r"License:\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        lic = m.group(1)
    doi = None
    m = DOI_RE.search(text)
    if m:
        doi = m.group(1)
    return lic, doi

def parse_license_and_doi_from_page(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    text = soup.get_text("\n", strip=True)
    lic, doi = parse_license_and_doi_from_text(text)
    custom_license_text = None
    # Only capture explicit LICENSE AGREEMENT blocks (if present on page)
    m = re.search(r"([A-Z][A-Z \d\.\-]*LICENSE AGREEMENT.*)", text, re.IGNORECASE|re.DOTALL)
    if m:
        custom_license_text = m.group(1)[:20000]
    return lic, doi, custom_license_text

def find_external_arxiv_from_page(soup: BeautifulSoup) -> Optional[str]:
    txt = soup.get_text("\n", strip=True)
    m = ARXIV_RE.search(txt)
    if m:
        return m.group(1)
    return None

def find_doi_and_arxiv_from_readme(readme_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Fallback: scan README for DOI (incl. doi.org links) and arXiv IDs."""
    doi = None
    m = DOI_RE.search(readme_text or "")
    if m:
        doi = m.group(1)
    arxiv = None
    m = ARXIV_RE.search(readme_text or "")
    if m:
        arxiv = m.group(1)
    return doi, arxiv

def find_pipeline_tag_from_api(model_info: Dict[str, Any]) -> Optional[str]:
    return model_info.get("pipeline_tag")

def list_model_files_from_api(model_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = []
    for f in model_info.get("siblings", []) or []:
        name = f.get("rfilename") or f.get("filename")
        if not name: continue
        entry = {"name": name}
        if "size" in f and isinstance(f["size"], int): entry["size"] = f["size"]
        if "lfs" in f and isinstance(f["lfs"], dict): entry["lfs"] = f["lfs"]
        files.append(entry)
    return files

def try_fetch_sha256_from_blob(repo_id: str, filename: str, commit_sha: Optional[str], timeout: int = 30) -> Optional[str]:
    ref = commit_sha or "main"
    url = f"{HF_BASE}/{repo_id}/blob/{quote(ref)}/{quote(filename)}"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200: return None
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
        h = hashlib.sha256(); total = 0
        for chunk in r.iter_content(16384):
            if not chunk: continue
            total += len(chunk)
            if total > max_bytes: return None
            h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def sha_from_lfs_meta(f: Dict[str, Any]) -> Optional[str]:
    lfs = f.get("lfs") or {}
    if not isinstance(lfs, dict): return None
    if isinstance(lfs.get("sha256"), str): return lfs["sha256"]
    oid = lfs.get("oid")
    if isinstance(oid, str):
        m = re.match(r"sha256:([0-9a-f]{64})$", oid)
        if m: return m.group(1)
    return None

def normalize_spdx_license(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return SPDX_LICENSE_CANON.get(s.strip().lower(), None)

def get_readme_text(model_info: Dict[str, Any], timeout: int = 30) -> str:
    model_id = model_info.get("modelId")
    for f in model_info.get("siblings", []) or []:
        name = f.get("rfilename") or f.get("filename")
        if name and name.lower() in {"readme.md","modelcard.md"}:
            raw = f"{HF_BASE}/{model_id}/resolve/{model_info.get('sha','main')}/{quote(name)}"
            try:
                rr = requests.get(raw, timeout=timeout)
                if rr.status_code == 200: return rr.text
            except Exception: pass
    try:
        r = requests.get(f"{HF_BASE}/{model_id}", timeout=timeout)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    except Exception: pass
    return ""

def try_read_license_file(repo_id: str, model_info: Dict[str, Any], timeout: int = 30) -> Optional[str]:
    commit_sha = model_info.get("sha") or "main"
    for f in model_info.get("siblings") or []:
        name = f.get("rfilename") or f.get("filename")
        if not name: continue
        low = name.lower()
        if low == "license" or low.startswith("license"):
            url = f"{HF_BASE}/{repo_id}/resolve/{commit_sha}/{quote(name)}"
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200 and r.text.strip(): return r.text
            except Exception: pass
    return None

def is_probably_small_file(name: str) -> bool:
    nlow = name.lower()
    if nlow in (s.lower() for s in SMALL_FILE_NAMES): return True
    for ext in SMALL_FILE_EXTS:
        if nlow.endswith(ext): return True
    return False

# -------- Dependencies (evidence-based only) --------
def parse_requirements_text(text: str) -> Set[str]:
    pkgs = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        m = re.match(r"([A-Za-z0-9_.\-]+)", line)
        if m:
            low = m.group(1).lower()
            if low == "opencv": low = "opencv-python"
            pkgs.add(low)
    return pkgs

def get_requirements_from_repo(repo_id: str, model_info: Dict[str, Any], timeout: int = 30) -> Set[str]:
    commit_sha = model_info.get("sha") or "main"
    found = set()
    for f in model_info.get("siblings") or []:
        name = (f.get("rfilename") or f.get("filename") or "").strip()
        if name in REQ_FILES:
            url = f"{HF_BASE}/{repo_id}/resolve/{commit_sha}/{quote(name)}"
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200 and r.text:
                    if name.lower().startswith("requirements"):
                        found |= parse_requirements_text(r.text)
                    elif name.lower().startswith("environment"):
                        deps = re.findall(r"-\s*([A-Za-z0-9_.\-]+)", r.text)
                        found |= {("opencv-python" if d.lower()=="opencv" else d.lower()) for d in deps}
                    elif name == "pyproject.toml":
                        pkgs = re.findall(r'^\s*([A-Za-z0-9_.\-]+)\s*=\s*".*"$', r.text, flags=re.MULTILINE)
                        found |= {("opencv-python" if p.lower()=="opencv" else p.lower()) for p in pkgs}
                    elif name in {"setup.cfg","setup.py","Pipfile"}:
                        pkgs = re.findall(r"(?:install_requires|requires).*\[?([^\]]+)\]?", r.text, flags=re.IGNORECASE)
                        for seg in pkgs:
                            for tok in seg.split(","):
                                tok = tok.strip().strip("'\"")
                                if tok:
                                    found.add(tok.split()[0].lower())
            except Exception:
                pass
    return found

# -------------- SPDX assembly --------------
def build_spdx(repo_id: str,
               model_info: Dict[str, Any],
               license_raw: Optional[str],
               custom_license_text: Optional[str],
               doi: Optional[str],
               arxiv_id: Optional[str],
               files: List[Dict[str, Any]],
               sha_map: Dict[str, Optional[str]],
               pipeline_tag: Optional[str],
               force_dataset: bool,
               include_dataset_details: bool,
               timeout: int) -> Dict[str, Any]:

    # Profiles
    card = model_info.get("cardData") or {}
    datasets_meta = card.get("datasets") if isinstance(card, dict) else None
    include_dataset_profile = bool(datasets_meta) or force_dataset

    profile = ["core","software","simpleLicensing","ai"]
    if include_dataset_profile:
        profile.append("dataset")

    commit_sha = model_info.get("sha") or "main"

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
                {"type": "Tool", "spdxId": "SPDXRef-TOOL", "name": "hf2spdx-ai-bom", "version": "1.8.0"}
            ]
        },
        "element": [],
        "rootElement": ["SPDXRef-MODEL"],
        "externalRef": []
    }

    # AIPackage (minimal)
    ai_pkg = {
        "type": "AIPackage",
        "spdxId": "SPDXRef-MODEL",
        "name": repo_id,
        "packageVersion": commit_sha,
        "downloadLocation": f"{HF_BASE}/{repo_id}/tree/{quote(commit_sha)}",
    }
    # optional description from pipeline_tag
    if pipeline_tag:
        ai_pkg["description"] = f"{pipeline_tag} model on Hugging Face"
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

    # Dependencies (only with explicit evidence)
    deps: Set[str] = set()
    deps |= get_requirements_from_repo(repo_id, model_info, timeout=timeout)

    readme_text = get_readme_text(model_info, timeout=timeout)
    deps |= extract_pkgs_from_readme(readme_text)

    if deps:
        for name in sorted(deps):
            pid = f"SPDXRef-PKG-{name.upper()}"
            pkg = {
                "type": "Package",
                "spdxId": pid,
                "name": name,
                "downloadLocation": f"https://pypi.org/project/{name}/",
                "packageUrl": f"pkg:pypi/{name}"
            }
            doc["element"].append(pkg)
            rels.append({
                "type": "Relationship",
                "spdxId": f"SPDXRef-REL-DEP-{name.upper()}",
                "from": "SPDXRef-MODEL",
                "to": pid,
                "relationshipType": "dependsOn"
            })

    # Licensing (declared & concluded)
    lic_std = normalize_spdx_license(license_raw)
    if lic_std:
        for sid, rtype in (("SPDXRef-LIC-MODEL-DECLARED","hasDeclaredLicense"),
                           ("SPDXRef-LIC-MODEL-CONCLUDED","hasConcludedLicense")):
            doc["element"].append({"type":"LicenseExpression","spdxId":sid,"licenseExpression":lic_std})
            rels.append({"type":"Relationship","spdxId":f"SPDXRef-REL-{sid.split('-')[-1]}",
                         "from":"SPDXRef-MODEL","to":sid,"relationshipType":rtype})
    else:
        custom_text = custom_license_text or (license_raw or "Custom license (text unavailable)")
        for sid, rtype in (("SPDXRef-LIC-MODEL-DECLARED","hasDeclaredLicense"),
                           ("SPDXRef-LIC-MODEL-CONCLUDED","hasConcludedLicense")):
            doc["element"].append({"type":"SimpleLicensingText","spdxId":sid,"licenseText":custom_text})
            rels.append({"type":"Relationship","spdxId":f"SPDXRef-REL-{sid.split('-')[-1]}",
                         "from":"SPDXRef-MODEL","to":sid,"relationshipType":rtype})

    # Dataset (only if cardData.datasets present or force)
    include_dataset_profile_flag = include_dataset_profile
    if include_dataset_profile_flag:
        ds_name = None
        if isinstance(datasets_meta, list) and datasets_meta:
            ds_name = ", ".join(map(str, datasets_meta[:3])) + ("..." if len(datasets_meta) > 3 else "")
        else:
            ds_name = "Not disclosed" if force_dataset else None

        if ds_name:
            ds = {
                "type": "DatasetPackage",
                "spdxId": "SPDXRef-DATASET-TRAIN",
                "name": ds_name
            }
            # optional details only if flag is set
            if include_dataset_details and pipeline_tag:
                pt = pipeline_tag.lower()
                if "image" in pt: ds["datasetType"] = "image"
                elif "audio" in pt: ds["datasetType"] = "audio"
                else: ds["datasetType"] = "text"

            doc["element"].append(ds)
            rels.append({
                "type": "Relationship",
                "spdxId": "SPDXRef-REL-MODEL-TRAINEDON",
                "from": "SPDXRef-MODEL",
                "to": "SPDXRef-DATASET-TRAIN",
                "relationshipType": "trainedOn"
            })

    # Describes + relationships
    doc["element"].append({"type":"Relationship","spdxId":"SPDXRef-REL-DESCRIBES",
                           "from":"SPDXRef-DOCUMENT","to":"SPDXRef-MODEL","relationshipType":"describes"})
    doc["element"].extend(rels)

    if doi:
        doc["externalRef"].append({"type":"ExternalRef","externalRefType":"doi","locator":doi})
    if arxiv_id:
        doc["externalRef"].append({"type":"ExternalRef","externalRefType":"arXiv","locator":arxiv_id})

    return doc

# -------------- Pipeline --------------
def main():
    ap = argparse.ArgumentParser(description="Generate SPDX 3.0 AI-BOM for a Hugging Face model (facts-only).")
    ap.add_argument("input", help="Hugging Face repo_id or URL")
    ap.add_argument("-o","--output", nargs="?", const="", help="Output file (default: <repo_id>.spdx3.json). If omitted, print to stdout.")
    ap.add_argument("--force-dataset", action="store_true", help="Force add placeholder DatasetPackage + trainedOn even if no datasets metadata")
    ap.add_argument("--dataset-details", action="store_true", help="Include optional dataset details (datasetType mapping from pipeline). OFF by default.")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    args = ap.parse_args()

    try:
        repo_id = extract_repo_id_from_url_or_id(args.input)
    except ValueError as e:
        print(f"[ERR] {e}", file=sys.stderr); sys.exit(1)

    try:
        model_info = get_model_info(repo_id, timeout=args.timeout)
    except Exception as e:
        print(f"[ERR] Failed to read HF API: {e}", file=sys.stderr); sys.exit(2)

    # License / DOI / LICENSE file content (all fetched from page / files)
    lic_raw = doi = lic_text = None
    arxiv_id = None
    readme_text = ""
    try:
        soup = get_model_page(repo_id, timeout=args.timeout)
        lic_raw, doi, lic_text_page = parse_license_and_doi_from_page(soup)
        if lic_text_page:
            lic_text = lic_text_page
        arxiv_id = find_external_arxiv_from_page(soup)
    except Exception:
        pass

    # Fallbacks: API / README scanning
    if not lic_raw:
        lic_raw = model_info.get("license")
    readme_text = get_readme_text(model_info, timeout=args.timeout)
    if not doi or not arxiv_id:
        doi_rd, arxiv_rd = find_doi_and_arxiv_from_readme(readme_text)
        if not doi:
            doi = doi_rd
        if not arxiv_id:
            arxiv_id = arxiv_rd

    lic_text_file = try_read_license_file(repo_id, model_info, timeout=args.timeout)
    if lic_text_file:
        lic_text = lic_text_file

    pipeline_tag = find_pipeline_tag_from_api(model_info)

    siblings = list_model_files_from_api(model_info)
    commit_sha = model_info.get("sha")

    # SHA collection (LFS -> blob page -> raw small files)
    sha_map: Dict[str, Optional[str]] = {}
    for f in siblings:
        fname = f["name"]
        sha_val = sha_from_lfs_meta(f)
        try:
            if not sha_val and (fname.endswith(tuple(BINARY_EXTS)) or (fname in COMMON_CONFIG_FILES)):
                sha_val = try_fetch_sha256_from_blob(repo_id, fname, commit_sha, timeout=args.timeout)
            size = f.get("size")
            if (not sha_val) and ((isinstance(size, int) and size <= RAW_HASH_MAX) or (size is None and is_probably_small_file(fname))):
                sha_val = try_compute_sha256_from_raw(repo_id, fname, commit_sha, max_bytes=RAW_HASH_MAX, timeout=args.timeout)
        except Exception:
            sha_val = None
        sha_map[fname] = sha_val

    spdx = build_spdx(
        repo_id=repo_id,
        model_info=model_info,
        license_raw=lic_raw,
        custom_license_text=lic_text,
        doi=doi,
        arxiv_id=arxiv_id,
        files=siblings,
        sha_map=sha_map,
        pipeline_tag=pipeline_tag,
        force_dataset=args.force_dataset,
        include_dataset_details=args.dataset_details,
        timeout=args.timeout
    )

    if args.output is not None:
        out = args.output if args.output else f"{repo_id.replace('/', '_')}.spdx3.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(spdx, f, ensure_ascii=False, indent=2)
        print(f"[OK] SPDX AI-BOM saved to: {out}")
    else:
        json.dump(spdx, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
