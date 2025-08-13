### hf2spdx-ai-bom 更新說明（v1.0 → v2.0.0）

此檔精簡自原 `update_note.md`，保留關鍵差異與版本要點。

## 版本重點

- v1.0 → v1.4：逐步加強「完整列檔、雜湊可得、授權更準、輸出可重現（commit 釘選）、適度推斷」。
- v1.5 → v2.0.0：回歸並深化「事實為本」，僅納入可直接取用之事實；逐版提升 README/頁面抽取魯棒性與雜訊過濾；v2.0.0 補齊元素 `creationInfo`、供應者、釋出時間等中繼資料。

## 重大差異一覽（節選）

| 面向 | v1.0..v1.4 | v1.5..v1.8.2 | v2.0.0 |
|---|---|---|---|
| AIPackage `downloadLocation` | 倉庫根目錄/`tree/<commit>` | `tree/<commit>` | `tree/<commit>` |
| File `downloadLocation` | `resolve/main/...` | `resolve/<commit>/...` | `resolve/<commit>/...` |
| 雜湊 | LFS/Blob/Raw（小檔） | 同左 | 同左 |
| Dataset | 可偵測/占位 | 僅 API 有或 `--force-dataset` | 同左（可加 `--dataset-details`） |
| ExternalRef |（無/DOI） | DOI + arXiv（含 README 後備） | 同左 |
| 中繼資料 | - | - | 元素 `creationInfo`、`AIPackage.primaryPurpose/releaseTime/suppliedBy` |

## 插件（enrichers）

- `spdx_hf_enricher.py`：從 HF API/README 抽取補強 `AIPackage` 欄位。
- `spdx_ailuminate_enricher.py`：從 MLCommons AILuminate v1.0 Bare Models 映射 `safetyRiskAssessment`（保留 `--add-comment` 與 `--force`）。

共通 CLI 行為：預設 dry‑run；`-o` 三態（`-o`=另存 `enriched.<原檔>`；`-o orig`=就地覆寫；或 `-o <檔名>`=單檔）。帶 `-o` 時列印「=== Write Summary ===」。

## 遷移指南

- 主腳本由 `hf2spdx-ai-bom_v2.0.0.py` 移至 `src/hf2spdx_ai_bom.py`。
- Enrichers 由 `plugin/` 改為 `src/enrichers/`，並保留 `src/enrichers/README.md`。
- 建議輸出目錄統一為 `output/samples/`；大量歷史輸出可移至 `archive/` 或由 `.gitignore` 排除。


