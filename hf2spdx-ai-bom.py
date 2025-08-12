#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hf2spdx-ai-bom.py
"""
import json, re, sys, time, argparse
from typing import Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

HF_BASE = "https://huggingface.co"
API_MODEL = HF_BASE + "/api/models/{repo_id}"

def extract_repo_id_from_url_or_id(input_str: str) -> str:
    """從 URL 或直接的 repo_id 中提取 repo_id"""
    # 如果輸入是 URL，則提取 repo_id
    if input_str.startswith(('http://', 'https://')):
        parsed = urlparse(input_str)
        if 'huggingface.co' in parsed.netloc:
            # 移除開頭的 '/' 並取得路徑的前兩部分（用戶名/模型名）
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                return f"{path_parts[0]}/{path_parts[1]}"
            else:
                raise ValueError(f"無效的 Hugging Face URL 格式: {input_str}")
        else:
            raise ValueError(f"非 Hugging Face URL: {input_str}")
    else:
        # 假設輸入已經是 repo_id 格式
        return input_str

def get_model_info(repo_id: str) -> Dict[str, Any]:
    # 盡量用 API（通用）；若速率/權限問題，頁面 fallback 也可
    url = API_MODEL.format(repo_id=repo_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def get_model_page(repo_id: str) -> BeautifulSoup:
    r = requests.get(f"{HF_BASE}/{repo_id}", timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def extract_license_and_doi_from_page(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    text = soup.get_text(" ", strip=True)
    # License
    lic = None
    m = re.search(r"License:\s*([^\s/]+)", text, re.IGNORECASE)
    if m:
        lic = m.group(1)
    # DOI
    doi = None
    m = re.search(r"\bdoi:([0-9./]+)", text, re.IGNORECASE)
    if m:
        doi = m.group(1)
    return {"license": lic, "doi": doi}

def find_pipeline_tag(soup: BeautifulSoup) -> Optional[str]:
    # 模型頁頂部通常顯示任務標籤，如 Text Generation
    txt = soup.get_text(" ", strip=True)
    if "Text Generation" in txt:
        return "text-generation"
    return None

def get_safetensors_sha256(repo_id: str) -> Optional[str]:
    # 嘗試從檔案頁「Raw pointer details」取得 SHA256（適用 Xet/LFS）
    r = requests.get(f"{HF_BASE}/{repo_id}/blob/main/model.safetensors", timeout=20)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    m = re.search(r"SHA256:\s*([0-9a-f]{64})", text, re.IGNORECASE)
    return m.group(1) if m else None

def build_spdx(repo_id: str,
               license_expr: Optional[str],
               doi: Optional[str],
               safetensors_sha256: Optional[str],
               pipeline_tag: Optional[str]) -> Dict[str, Any]:
    # 某些欄位若取不到，用 None 或合宜的占位
    doc = {
      "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
      "type": "SpdxDocument",
      "spdxId": "SPDXRef-DOCUMENT",
      "name": f"AI-BOM for {repo_id}",
      "profileConformance": ["core", "software", "simpleLicensing", "ai", "dataset"],
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
          { "type": "Organization", "spdxId": "SPDXRef-ORG", "name": "Hugging Face Hub Scraper (example)" }
        ],
        "createdUsing": [
          { "type": "Tool", "spdxId": "SPDXRef-TOOL", "name": "hf2spdx-ai-bom", "version": "1.0.0" }
        ]
      },
      "element": [],
      "rootElement": ["SPDXRef-MODEL"],
      "externalRef": []
    }

    # AIPackage
    ai_pkg = {
      "type": "AIPackage",
      "spdxId": "SPDXRef-MODEL",
      "name": repo_id,
      "description": f"Model from {repo_id}",
      "packageVersion": "main",
      "downloadLocation": f"{HF_BASE}/{repo_id}",
      "standardCompliance": [],
      "typeOfModel": "Transformer",
      "domain": ["NLP"],
      "hyperparameter": {},
      "modelExplainability": None,
      "informationAboutTraining": "Information not specified",
      "modelDataPreprocessing": "Not specified",
      "metric": [],
      "safetyRiskAssessment": [],
      "useSensitivePersonalInformation": None,
      "energyConsumption": []
    }
    doc["element"].append(ai_pkg)

    # File (model.safetensors)
    file_elem = {
      "type": "File",
      "spdxId": "SPDXRef-MODEL-FILE",
      "name": "model.safetensors",
      "fileKind": "binary",
      "hash": [],
      "downloadLocation": f"{HF_BASE}/{repo_id}/resolve/main/model.safetensors"
    }
    if safetensors_sha256:
        file_elem["hash"].append({"type": "Hash", "algorithm": "SHA256", "hashValue": safetensors_sha256})
    doc["element"].append(file_elem)

    # Dependency example（transformers）
    dep = {
      "type": "Package",
      "spdxId": "SPDXRef-PKG-TRANSFORMERS",
      "name": "transformers",
      "packageVersion": None,
      "downloadLocation": "https://pypi.org/project/transformers/",
      "packageUrl": "pkg:pypi/transformers"
    }
    doc["element"].append(dep)

    # Licenses
    lic_expr = license_expr or "NOASSERTION"
    for sid in ("SPDXRef-LIC-MODEL-DECLARED", "SPDXRef-LIC-MODEL-CONCLUDED"):
        doc["element"].append({ "type": "LicenseExpression", "spdxId": sid, "licenseExpression": lic_expr })

    # Relationships
    rels = [
      {"type":"Relationship","spdxId":"SPDXRef-REL-DOC-DESCRIBES","from":"SPDXRef-DOCUMENT","to":"SPDXRef-MODEL","relationshipType":"describes"},
      {"type":"Relationship","spdxId":"SPDXRef-REL-MODEL-CONTAINS-FILE","from":"SPDXRef-MODEL","to":"SPDXRef-MODEL-FILE","relationshipType":"contains"},
      {"type":"Relationship","spdxId":"SPDXRef-REL-MODEL-DEPENDS-TRANSFORMERS","from":"SPDXRef-MODEL","to":"SPDXRef-PKG-TRANSFORMERS","relationshipType":"dependsOn"},
      {"type":"Relationship","spdxId":"SPDXRef-REL-MODEL-DECLARED-LIC","from":"SPDXRef-MODEL","to":"SPDXRef-LIC-MODEL-DECLARED","relationshipType":"hasDeclaredLicense"},
      {"type":"Relationship","spdxId":"SPDXRef-REL-MODEL-CONCLUDED-LIC","from":"SPDXRef-MODEL","to":"SPDXRef-LIC-MODEL-CONCLUDED","relationshipType":"hasConcludedLicense"}
    ]
    doc["element"].extend(rels)

    # External refs (DOI)
    if doi:
        doc["externalRef"].append({"type":"ExternalRef","externalRefType":"doi","locator":doi})

    return doc

def main():
    parser = argparse.ArgumentParser(description="為 Hugging Face 模型生成 SPDX 3.0 AI-BOM")
    parser.add_argument("input", help="Hugging Face 存儲庫 ID（例如：microsoft/DialoGPT-medium）或完整 URL（例如：https://huggingface.co/microsoft/DialoGPT-medium）")
    parser.add_argument("-o", "--output", nargs="?", const="", help="輸出檔案（預設：<repo_id>.spdx3.json）")
    
    args = parser.parse_args()
    
    # 從輸入中提取 repo_id
    try:
        repo_id = extract_repo_id_from_url_or_id(args.input)
    except ValueError as e:
        print(f"錯誤：{e}", file=sys.stderr)
        sys.exit(1)

    # 決定輸出檔案名稱
    if args.output is not None:  # 有指定 -o 參數
        if args.output == "":  # -o 後面沒加檔名，使用預設
            output_file = f"{repo_id.replace('/', '_')}.spdx3.json"
        else:  # -o 後面有指定檔名
            output_file = args.output
    else:  # 沒有 -o 參數，輸出到 stdout
        output_file = None

    # 1) 嘗試 API 取模型資訊（通用）
    try:
        _ = get_model_info(repo_id)  # 保留進一步擴充用；本例主要用頁面抓 DOI/License
    except Exception:
        pass

    # 2) 頁面抓 license / doi / pipeline tag
    lic = doi = pipe = None
    try:
        soup = get_model_page(repo_id)
        meta = extract_license_and_doi_from_page(soup)
        lic = meta.get("license")
        doi = meta.get("doi")
        pipe = find_pipeline_tag(soup)
    except Exception:
        pass

    # 3) 取 model.safetensors 的 SHA256（若可）
    sha256 = None
    try:
        sha256 = get_safetensors_sha256(repo_id)
    except Exception:
        pass

    # 4) 組裝 SPDX 3.0 JSON-LD
    spdx = build_spdx(repo_id, lic, doi, sha256, pipe)
    
    # 5) 輸出結果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(spdx, f, ensure_ascii=False, indent=2)
        print(f"SPDX AI-BOM saved to: {output_file}")
    else:
        json.dump(spdx, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
