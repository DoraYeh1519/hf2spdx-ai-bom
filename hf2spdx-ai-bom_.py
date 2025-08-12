#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hf2spdx-ai-bom.py (enhanced)
- 以 Hugging Face API + 頁面爬取產生 SPDX 3.0 (JSON-LD) AI-BOM
- 重要改進：
  1) 為「所有檔案」建立 File + contains，並盡力抓取 SHA256
  2) 常見周邊檔 (config/tokenizer/generation_config/index 等) 一律納入
  3) dataset profile 嚴格一致：自動偵測；必要時可 --force-dataset
  4) useSensitivePersonalInformation → bool (預設 False)
  5) License 正規化為 SPDX 短代號
"""
import json, re, sys, time, argparse
from typing import Optional, Dict, Any, List, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote

HF_BASE = "https://huggingface.co"
API_MODEL = HF_BASE + "/api/models/{repo_id}"

# 常見 license 名稱到 SPDX 標準短代號的簡易對照（可自行擴充）
SPDX_LICENSE_CANON = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache2": "Apache-2.0",
    "apache2.0": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "gpl-3.0": "GPL-3.0-only",
    "lgpl-3.0": "LGPL-3.0-only",
    "cc-by-4.0": "CC-BY-4.0",
    "cc0-1.0": "CC0-1.0",
}

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

# ---- 抓資料 ---------------------------------------------------------------
def extract_repo_id_from_url_or_id(input_str: str) -> str:
    if input_str.startswith(('http://', 'https://')):
        parsed = urlparse(input_str)
        if 'huggingface.co' in parsed.netloc:
            parts = [p for p in parsed.path.strip('/').split('/') if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            raise ValueError(f"無效的 Hugging Face URL: {input_str}")
        raise ValueError(f"非 Hugging Face URL: {input_str}")
    return input_str

def get_model_info(repo_id: str) -> Dict[str, Any]:
    url = API_MODEL.format(repo_id=repo_id)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def get_model_page(repo_id: str) -> BeautifulSoup:
    r = requests.get(f"{HF_BASE}/{repo_id}", timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def parse_license_and_doi_from_page(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    text = soup.get_text(" ", strip=True)
    lic = None
    m = re.search(r"License:\s*([A-Za-z0-9.\-+]+)", text, re.IGNORECASE)
    if m:
        lic = m.group(1)
    doi = None
    m = re.search(r"\bdoi:\s*([0-9.]+/[A-Za-z0-9._/-]+)", text, re.IGNORECASE)
    if m:
        doi = m.group(1)
    return lic, doi

def find_pipeline_tag_from_api(model_info: Dict[str, Any]) -> Optional[str]:
    # API 欄位通常為 pipeline_tag 或 tags 中包含任務
    if isinstance(model_info, dict):
        if model_info.get("pipeline_tag"):
            return model_info["pipeline_tag"]
        tags = model_info.get("tags") or []
        # 嘗試從 tags 猜任務（保守作法）
        for t in tags:
            if t in {"text-generation","fill-mask","token-classification","image-classification"}:
                return t
    return None

def list_model_files_from_api(model_info: Dict[str, Any]) -> List[str]:
    files = []
    for f in model_info.get("siblings", []) or []:
        name = f.get("rfilename") or f.get("filename")
        if name:
            files.append(name)
    return files

def try_fetch_sha256_from_blob(repo_id: str, filename: str) -> Optional[str]:
    # 走 blob 頁面抓 "Raw pointer details" 的 SHA256（若 Xet/LFS 有顯示）
    url = f"{HF_BASE}/{repo_id}/blob/main/{quote(filename)}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    m = re.search(r"SHA256:\s*([0-9a-f]{64})", text, re.IGNORECASE)
    return m.group(1) if m else None

def normalize_spdx_license(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    key = s.strip().lower()
    return SPDX_LICENSE_CANON.get(key, s)  # 若未知就直接回傳原字串（交由工具判斷）

def infer_dataset_presence(readme_text: str) -> bool:
    # 很保守的啟發式：若 README 提到 datasets 關鍵詞就認定有資料集敘述
    kws = [
        "dataset", "datasets", "webtext", "c4", "pile", "common crawl",
        "pretrain", "fine-tune", "fine tune", "training data"
    ]
    low = readme_text.lower()
    return any(k in low for k in kws)

def get_readme_text_from_api(model_info: Dict[str, Any]) -> str:
    # API 不一定直接回 README；若 siblings 有 README.md 再抓 raw
    for f in model_info.get("siblings", []) or []:
        name = f.get("rfilename") or f.get("filename")
        if name and name.lower() in {"readme.md", "modelcard.md"}:
            # 讀 raw
            raw = f"{HF_BASE}/{model_info['modelId']}/resolve/main/{quote(name)}"
            try:
                rr = requests.get(raw, timeout=30)
                if rr.status_code == 200:
                    return rr.text
            except Exception:
                pass
    # 退一步，用主頁文字
    try:
        soup = get_model_page(model_info["modelId"])
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""

# ---- SPDX 組裝 -----------------------------------------------------------
def build_spdx(repo_id: str,
               model_info: Dict[str, Any],
               license_expr: Optional[str],
               doi: Optional[str],
               files: List[str],
               sha_map: Dict[str, Optional[str]],
               pipeline_tag: Optional[str],
               force_dataset: bool,
               validate_minimal: bool) -> Dict[str, Any]:

    profile = ["core", "software", "simpleLicensing", "ai"]
    readme_text = get_readme_text_from_api(model_info)
    has_dataset_info = infer_dataset_presence(readme_text)
    if has_dataset_info or force_dataset:
        profile.append("dataset")

    # SPDX 文件骨架
    doc = {
        "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
        "type": "SpdxDocument",
        "spdxId": "SPDXRef-DOCUMENT",
        "name": f"AI-BOM for {repo_id}",
        "profileConformance": profile,
        "dataLicense": {
            "type": "SimpleLicensingText",
            "spdxId": "SPDXRef-DATA-LICENSE",
            "licenseText": "CC0-1.0"
        },
        "creationInfo": {
            "type": "CreationInfo",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "specVersion": "3.0.1",
            "createdBy": [
                {"type": "Organization", "spdxId": "SPDXRef-ORG", "name": "Hugging Face Hub Scraper (example)"}
            ],
            "createdUsing": [
                {"type": "Tool", "spdxId": "SPDXRef-TOOL", "name": "hf2spdx-ai-bom", "version": "1.1.0"}
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
        "packageVersion": model_info.get("sha") or "main",
        "downloadLocation": f"{HF_BASE}/{repo_id}",
        "standardCompliance": [],
        "typeOfModel": "Transformer",
        "domain": ["NLP"],
        "hyperparameter": {},
        "modelExplainability": None,
        "informationAboutTraining": "Not specified" if not has_dataset_info else "See model card",
        "modelDataPreprocessing": "Not specified",
        "metric": [],
        "safetyRiskAssessment": [],
        "useSensitivePersonalInformation": False,  # ← 布林，避免空值
        "energyConsumption": []
    }
    doc["element"].append(ai_pkg)

    # Files（所有檔案都列入；重要檔案會自動覆蓋）
    rels: List[Dict[str, Any]] = []
    for idx, fname in enumerate(sorted(set(files))):
        spdx_id = f"SPDXRef-FILE-{idx:04d}"
        file_kind = "binary" if fname.endswith((".safetensors", ".bin", ".onnx")) else "text"
        hashes = []
        sha = sha_map.get(fname)
        if sha:
            hashes.append({"type": "Hash", "algorithm": "SHA256", "hashValue": sha})

        doc["element"].append({
            "type": "File",
            "spdxId": spdx_id,
            "name": fname,
            "fileKind": file_kind,
            "hash": hashes,
            "downloadLocation": f"{HF_BASE}/{repo_id}/resolve/main/{quote(fname)}"
        })

        rels.append({
            "type": "Relationship",
            "spdxId": f"SPDXRef-REL-CONTAINS-{idx:04d}",
            "from": "SPDXRef-MODEL",
            "to": spdx_id,
            "relationshipType": "contains"
        })

    # Dependency（盡力標：transformers）
    doc["element"].append({
        "type": "Package",
        "spdxId": "SPDXRef-PKG-TRANSFORMERS",
        "name": "transformers",
        "packageVersion": None,
        "downloadLocation": "https://pypi.org/project/transformers/",
        "packageUrl": "pkg:pypi/transformers"
    })
    rels.append({
        "type": "Relationship",
        "spdxId": "SPDXRef-REL-DEP-TRANSFORMERS",
        "from": "SPDXRef-MODEL",
        "to": "SPDXRef-PKG-TRANSFORMERS",
        "relationshipType": "dependsOn"
    })

    # Licenses（正規化）
    lic_expr = normalize_spdx_license(license_expr) or "NOASSERTION"
    for sid in ("SPDXRef-LIC-MODEL-DECLARED", "SPDXRef-LIC-MODEL-CONCLUDED"):
        doc["element"].append({
            "type": "LicenseExpression",
            "spdxId": sid,
            "licenseExpression": lic_expr
        })
        rels.append({
            "type": "Relationship",
            "spdxId": f"SPDXRef-REL-LIC-{sid.split('-')[-1]}",
            "from": "SPDXRef-MODEL",
            "to": sid,
            "relationshipType": "hasDeclaredLicense" if "DECLARED" in sid else "hasConcludedLicense"
        })

    # Dataset（有資訊或強制才加入，並補關聯）
    dataset_added = False
    if has_dataset_info or force_dataset:
        ds = {
            "type": "DatasetPackage",
            "spdxId": "SPDXRef-DATASET-TRAIN",
            "name": "Not disclosed" if force_dataset and not has_dataset_info else "See model card",
            "datasetType": "text",
            "intendedUse": "Pretraining",
            "datasetAvailability": "not-disclosed" if force_dataset and not has_dataset_info else "unspecified"
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

    # Relationships（包含 describes 與 contains/dependsOn 等）
    doc["element"].append({
        "type": "Relationship",
        "spdxId": "SPDXRef-REL-DESCRIBES",
        "from": "SPDXRef-DOCUMENT",
        "to": "SPDXRef-MODEL",
        "relationshipType": "describes"
    })
    doc["element"].extend(rels)

    # External refs: DOI
    if doi:
        doc["externalRef"].append({"type":"ExternalRef","externalRefType":"doi","locator":doi})

    # 最小一致性驗證：若 profile 有 dataset 但沒有 DatasetPackage，補一個佔位
    if validate_minimal and ("dataset" in doc["profileConformance"]) and (not dataset_added):
        ds = {
            "type": "DatasetPackage",
            "spdxId": "SPDXRef-DATASET-PLACEHOLDER",
            "name": "Not disclosed",
            "datasetType": "text",
            "intendedUse": "Unspecified",
            "datasetAvailability": "not-disclosed"
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

# ---- 主流程 ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="為 Hugging Face 模型生成 SPDX 3.0 AI-BOM（自動補齊檔案/雜湊/一致性）")
    p.add_argument("input", help="Hugging Face repo_id 或 URL，例如 openai-community/gpt2 或 https://huggingface.co/openai-community/gpt2")
    p.add_argument("-o", "--output", nargs="?", const="", help="輸出檔案（預設：<repo_id>.spdx3.json；不帶 -o 則輸出到 stdout）")
    p.add_argument("--force-dataset", action="store_true", help="即使讀不到資料集資訊，也強制加入占位 DatasetPackage 與 trainedOn 關聯")
    p.add_argument("--validate-minimal", action="store_true", help="最小一致性：若宣告 dataset profile 但未加入 DatasetPackage，則自動補一個占位")
    p.add_argument("--timeout", type=int, default=30, help="HTTP 逾時（秒）")
    args = p.parse_args()

    try:
        repo_id = extract_repo_id_from_url_or_id(args.input)
    except ValueError as e:
        print(f"[ERR] {e}", file=sys.stderr)
        sys.exit(1)

    # 1) 取 API
    try:
        model_info = get_model_info(repo_id)
    except Exception as e:
        print(f"[ERR] 讀取 API 失敗：{e}", file=sys.stderr)
        sys.exit(2)

    # 2) License / DOI（頁面解析 + API 備援）
    lic, doi = None, None
    try:
        soup = get_model_page(repo_id)
        lic, doi = parse_license_and_doi_from_page(soup)
    except Exception:
        pass
    if not lic:
        lic = model_info.get("license")
    lic = normalize_spdx_license(lic)

    # 3) Pipeline tag
    pipeline_tag = find_pipeline_tag_from_api(model_info)

    # 4) 檔案清單（API）與 SHA256（逐檔嘗試抓 blob 的 Raw pointer details）
    files = list_model_files_from_api(model_info)
    # 若倉庫沒有標準單檔 safetensors，COMMON_CONFIG_FILES 仍會被納入（由 API 提供）
    sha_map: Dict[str, Optional[str]] = {}
    for fname in files:
        sha = None
        try:
            # 只對較大的/二進位檔與 index 檔嘗試抓 SHA256，避免太多請求
            if fname.endswith((".safetensors", ".bin", ".onnx", ".pt", ".ckpt", ".index.json")) or fname in COMMON_CONFIG_FILES:
                sha = try_fetch_sha256_from_blob(repo_id, fname)
        except Exception:
            sha = None
        sha_map[fname] = sha

    # 5) 組裝 SPDX
    spdx = build_spdx(
        repo_id=repo_id,
        model_info=model_info,
        license_expr=lic,
        doi=doi,
        files=files,
        sha_map=sha_map,
        pipeline_tag=pipeline_tag,
        force_dataset=args.force_dataset,
        validate_minimal=args.validate_minimal
    )

    # 6) 輸出
    if args.output is not None:
        out = args.output if args.output else f"{repo_id.replace('/', '_')}.spdx3.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(spdx, f, ensure_ascii=False, indent=2)
        print(f"[OK] SPDX AI-BOM saved to: {out}")
    else:
        json.dump(spdx, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
