
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# hf2spdx-ai-bom.py (v1.6.1)
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

ARXIV_RE = re.compile(r"\barxiv\s*[:=]\s*([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.IGNORECASE)

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
OPTIONS_TAKING_VALUE = {"-i","--index-url","--extra-index-url","--find-links","--index-strategy"}

IMPORT_ALLOWLIST = {
    "transformers","torch","tensorflow","keras","flax","accelerate","diffusers",
    "onnxruntime","pillow","numpy","matplotlib","scipy","safetensors","bitsandbytes",
    "sentencepiece","tokenizers","timm","opencv-python","opencv","gradio","datasets",
    "peft","trl","vllm","mlx","kernels","pydantic"
}

def find_external_arxiv_from_page(soup: BeautifulSoup) -> Optional[str]:
    txt = soup.get_text("\n", strip=True)
    m = ARXIV_RE.search(txt)
    if m:
        return m.group(1)
    return None

def normalize_install_line_tail(tail: str) -> str:
    # Remove continuation backslashes and trailing comments
    t = tail.replace("\\\n", " ").replace("\\", " ")
    t = re.sub(r"\s+#.*$", "", t)
    return t

def extract_pkgs_from_install_tail(tail: str) -> Set[str]:
    pkgs: Set[str] = set()
    t = normalize_install_line_tail(tail)
    # Remove options-with-arg segments entirely to avoid leaking their values as "packages"
    t = re.sub(r'(?:^|\s)(?:-i|--index-url|--extra-index-url|--find-links|--index-strategy)\s+\S+', ' ', t)
    tokens = INSTALL_SPLIT_RE.split(t.strip())
    for tok in tokens:
        if not tok or tok in BAD_TOKENS: 
            continue
        if tok.startswith("-"): 
            continue
        low = tok.lower().strip().strip(",")
        # filter vcs/urls/files/wheels
        if "://" in low or low.startswith("git+") or low.startswith("file:") or low.endswith((".whl",".zip",".tar.gz",".tgz")):
            continue
        # extras: package[extra]=> package
        low = re.sub(r"\[.*\]$", "", low)
        # version pins/spaces: pkg==1.2.3 or pkg>=
        low = re.split(r"[<>=!~]", low)[0]
        # pure package pattern
        if not re.match(r"^[a-z0-9][a-z0-9._-]*$", low):
            continue
        # common false positives
        if low in {"git","https","http","pip","uv","python","mamba","conda"}:
            continue
        pkgs.add(low)
    return pkgs

def get_requirements_from_readme(readme_text: str) -> Set[str]:
    pkgs = set()
    # 1) Parse installer lines
    for m in INSTALL_CMD_RE.finditer(readme_text):
        tail = m.group("tail")
        pkgs |= extract_pkgs_from_install_tail(tail)
    # 2) Parse import lines (allowlist only)
    for m in re.finditer(r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+))",
                         readme_text, flags=re.IGNORECASE|re.MULTILINE):
        mod = (m.group(1) or m.group(2) or "").split(".")[0].lower()
        if mod in IMPORT_ALLOWLIST:
            # normalize opencv variants
            if mod == "opencv": mod = "opencv-python"
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

def parse_license_and_doi_from_page(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    text = soup.get_text("\n", strip=True)
    lic = None
    m = re.search(r"License:\s*([A-Za-z0-9.\-+]+)", text, re.IGNORECASE);  lic = m.group(1) if m else None
    doi = None
    m = re.search(r"\bdoi:\s*([0-9.]+/[A-Za-z0-9._/-]+)", text, re.IGNORECASE);  doi = m.group(1) if m else None
    custom_license_text = None
    # Only capture explicit LICENSE AGREEMENT blocks (if present on page)
    m = re.search(r"([A-Z][A-Z \d\.\-]*LICENSE AGREEMENT.*)", text, re.IGNORECASE|re.DOTALL)
    if m:
        custom_license_text = m.group(1)[:20000]
    return lic, doi, custom_license_text

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
        # pkg[extra]==ver ; pkg>=ver ; pkg
        m = re.match(r"([A-Za-z0-9_.\-]+)", line)
        if m: pkgs.add(m.group(1).lower())
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
                        # very rough YAML parse (only pip/conda deps in a list)
                        deps = re.findall(r"-\s*([A-Za-z0-9_.\-]+)", r.text)
                        found |= {d.lower() for d in deps}
                    elif name == "pyproject.toml":
                        pkgs = re.findall(r'^\s*([A-Za-z0-9_.\-]+)\s*=\s*".*"$', r.text, flags=re.MULTILINE)
                        found |= {p.lower() for p in pkgs}
                    elif name in {"setup.cfg","setup.py","Pipfile"}:
                        # Best effort: pick common library names
                        pkgs = re.findall(r"(?:install_requires|requires).*\[?([^\]]+)\]?", r.text, flags=re.IGNORECASE)
                        for seg in pkgs:
                            for tok in seg.split(","):
                                tok = tok.strip().strip("'\"")
                                if tok:
                                    found.add(tok.split()[0].lower())
            except Exception:
                pass
    return found


# (removed fallback get_requirements_from_readme)
