### hf2spdx-ai-bom 更新說明（v1.0 → v1.5）

#### 腳本用途
- 產生針對 Hugging Face 模型的 SPDX 3.0（JSON-LD）AI-BOM，彙整模型與相關檔案、授權、雜湊、相依套件、資料集註記等可追溯性資訊。

#### 基本用法
- v1.0：
  ```bash
  python hf2spdx-ai-bom_v1.0.py <repo_id_or_url> [-o out.json]
  ```
- v1.1：
  ```bash
  python hf2spdx-ai-bom_v1.1.py <repo_id_or_url> [-o out.json] [--force-dataset] [--validate-minimal] [--timeout 30]
  ```
- v1.2：
  ```bash
  python hf2spdx-ai-bom_v1.2.py <repo_id_or_url> [-o out.json] [--force-dataset] [--validate-minimal] [--timeout 30]
  ```
- v1.3：
  ```bash
  python hf2spdx-ai-bom_v1.3.py <repo_id_or_url> [-o out.json] [--force-dataset] [--validate-minimal] [--timeout 30]
  ```
- v1.4：
  ```bash
  python hf2spdx-ai-bom_v1.4.py <repo_id_or_url> [-o out.json] [--force-dataset] [--validate-minimal] [--timeout 30]
  ```
- v1.5（事實為本）：
  ```bash
  python hf2spdx-ai-bom_v1.5.py <repo_id_or_url> [-o out.json] [--force-dataset] [--dataset-details] [--timeout 30]
  ```

說明：
- `<repo_id_or_url>`：例如 `openai-community/gpt2` 或 `https://huggingface.co/openai-community/gpt2`。
- `-o/--output`：若不帶檔名，預設輸出成 `<repo_id>.spdx3.json`；未提供 `-o` 則輸出到 stdout。
- v1.1～v1.4：提供 `--force-dataset`、`--validate-minimal`、`--timeout`。
- v1.5：改為 `--force-dataset`、`--dataset-details`、`--timeout`；不再提供 `--validate-minimal`。

---

### 版本重點

#### v1.0（`hf2spdx-ai-bom_v1.0.py`）
- 以 HF API + 頁面擷取：
  - 嘗試讀取 License、DOI；任務標籤以頁面文字啟發式偵測。
  - 只建立單一 `File`：`model.safetensors`（若存在），`downloadLocation` 指向 `resolve/main/...`。
  - 嘗試從檔案頁的「Raw pointer details」擷取 `SHA256`（若可）。
- 相依：固定加入 `transformers` 1 個 `Package`。
- 授權：
  - 模型授權以 `LicenseExpression` 節點（若抓不到則 `NOASSERTION`）。
  - `dataLicense` 以 `SimpleLicensingText`（`licenseText: CC0-1.0`）。
- 其他欄位：`useSensitivePersonalInformation` 可能為 `None`；`fileKind` 使用 `binary`。
- 關聯：`describes`、`contains`、`dependsOn`、`hasDeclaredLicense`、`hasConcludedLicense`。

#### v1.1（`hf2spdx-ai-bom_v1.1.py`）
- 檔案覆蓋面：
  - 從 API 取得「所有檔案」，逐一建立 `File` 與 `contains` 關聯。
  - 嘗試為大型/二進位檔與常見設定檔（如 `config.json`、`tokenizer.json`、`generation_config.json` 等）抓取 `SHA256`（blob 頁面）。
  - `downloadLocation` 仍指向 `resolve/main/...`。
- 授權正規化：
  - 將常見 license 名稱正規化為 SPDX 短代號（如 `mit` → `MIT`）。
  - `LicenseExpression` 以正規化後的值呈現（抓不到則 `NOASSERTION`）。
  - `dataLicense` 仍為 `SimpleLicensingText`（`CC0-1.0`）。
- Dataset profile：
  - 解析 README 文字以偵測是否含資料集訊號；有則加入 `dataset` profile 與對應 `DatasetPackage`，並加上 `trainedOn` 關聯。
  - 提供 `--force-dataset` 強制加入占位 `DatasetPackage`；`--validate-minimal` 可在 profile 宣告但未加元素時自動補上占位。
- 其他：
  - `useSensitivePersonalInformation` 固定為 `False`（避免空值）。
  - `fileKind` 使用 `binary` 或 `text`。
  - 任務標籤改從 API 偵測（`pipeline_tag`）。
  - `AIPackage.packageVersion` 會寫入 API 回傳之 `sha`（若有），否則 `main`。

#### v1.2（`hf2spdx-ai-bom_v1.2.py`）
- 可重現性與雜湊：
  - 下載位址全面「釘選到 commit」：`downloadLocation` 改為 `resolve/<commit_sha>/...`。
  - 先嘗試從 blob 頁面抓 `SHA256`；若仍無且檔案較小（≤ 5 MB），則直接下載 raw 計算 `SHA256`。
  - `fileKind` 固定為 SPDX 3.0 建議的 `file`。
- 授權處理更嚴謹：
  - 若能正規化成 SPDX ID → 使用 `LicenseExpression`。
  - 否則建立 `SimpleLicensingText`，並嘗試擷取頁面中的「自訂/社群授權全文片段」（如 Llama 社群授權），納入 `licenseText`。
  - `dataLicense` 改為 `LicenseExpression("CC0-1.0")`（不再使用 `SimpleLicensingText`）。
- Dataset profile：維持 v1.1 行為（偵測或 `--force-dataset`），移除不支援的 `datasetAvailability` 值。
- 相依偵測：
  - 解析 README 以推斷 `transformers` 最低版本（若可），寫入 `sourceInfo`。
  - 額外偵測常見執行時依賴：`onnxruntime`、`pillow`、`torch`，自動加入 `Package` 與 `dependsOn`。

#### v1.3（`hf2spdx-ai-bom_v1.3.py`）
- 延續 v1.2 的 commit 釘選與 SHA 計算；新增：
  - 優先使用 LFS metadata 提供的 `sha256`，其次 blob 頁面，再退回小檔 raw 計算。
  - 嘗試讀取 `config.json` 以推斷 `typeOfModel` 與 `domain`，並結合 `pipeline_tag`。
  - Dataset：若 API `cardData.datasets` 有值則直接使用；否則仍以 README 偵測為輔，並支援占位元素（`--force-dataset`、`--validate-minimal`）。
  - 授權：若非 SPDX，優先採用倉庫中的 LICENSE 檔案內容作為 `SimpleLicensingText`。

#### v1.4（`hf2spdx-ai-bom_v1.4.py`）
- AIPackage 的 `downloadLocation` 改為 `.../tree/<commit>`（避免主分支漂移）。
- 小檔雜湊：即使 size 欄位缺失，也會依副檔名/檔名啟發式嘗試 raw 計算。
- `typeOfModel` 判斷更強（如 LLAMA → Decoder-only Transformer，ViT → Vision Transformer）。
- Dataset：`intendedUse` 依 `pipeline_tag`/repo/README 做更細緻的預設（如 text-generation → Pretraining / Fine-tuning）。
- 其他：沿用 LFS→blob→raw 的 SHA 取得流程與授權處理策略。

#### v1.5（`hf2spdx-ai-bom_v1.5.py`｜事實為本）
- 政策轉向「僅納入可直接從 HF API/頁面/檔案取得的事實」。
  - 不再推斷 `typeOfModel`/`domain` 等；除非來源明確，否則省略。
  - Dataset 僅在 API `cardData.datasets` 存在時才加入（或使用 `--force-dataset` 明示要求）。預設只寫入名稱，細節以 `--dataset-details` 額外開啟（僅做 pipeline→datasetType 的簡單映射）。
  - 相依僅接受「明確證據」：
    1) 倉庫中的 `requirements*.txt`、`environment.yml/.yaml`、`Pipfile`、`pyproject.toml`、`setup.cfg/py` 等；
    2) README 中可辨識的 `pip install X` 或程式碼片段的 `import X`。
  - 仍維持檔案與 AIPackage 的 commit 釘選 `downloadLocation`。
- 授權：若可正規化為 SPDX ID → `LicenseExpression`；否則使用 LICENSE 檔或頁面擷取之文字作為 `SimpleLicensingText`。
- 介面調整：新增 `--dataset-details`，不再提供 `--validate-minimal`。

---

### 重大差異一覽

| 面向 | v1.0 | v1.1 | v1.2 | v1.3 | v1.4 | v1.5 |
|---|---|---|---|---|---|---|
| 檔案列示 | 只含 `model.safetensors`（若可） | 列出 API 中所有檔案 | 同 v1.1 | 同 v1.1 | 同 v1.1 | 同 v1.1 |
| File `downloadLocation` | `resolve/main/...` | `resolve/main/...` | `resolve/<commit_sha>/...` | `resolve/<commit_sha>/...` | `resolve/<commit_sha>/...` | `resolve/<commit_sha>/...` |
| AIPackage `downloadLocation` | 倉庫根目錄 | 倉庫根目錄 | 倉庫根目錄 | 倉庫根目錄 | `.../tree/<commit>` | `.../tree/<commit>` |
| 雜湊（SHA256） | `model.safetensors` blob | 大型/常見檔案自 blob | blob→小檔 raw | LFS→blob→小檔 raw | LFS→blob→小檔 raw（含 size 缺失之啟發式） | 同 v1.4 |
| `fileKind` | `binary` | `binary`/`text` | `file` | `file` | `file` | `file` |
| 模型授權 | `LicenseExpression`/`NOASSERTION` | `LicenseExpression`（SPDX） | SPDX→`LicenseExpression`；否則 `SimpleLicensingText`（頁面片段） | 同 v1.2，但偏好 LICENSE 檔 | 同 v1.3 | 同 v1.3 |
| `dataLicense` | `SimpleLicensingText`（`CC0-1.0`） | 同 v1.0 | `LicenseExpression("CC0-1.0")` | 同 v1.2 | 同 v1.2 | 同 v1.2 |
| Dataset profile | 一律包含 | 偵測/旗標，可占位 | 同 v1.1（移除不支援值） | `cardData.datasets` 優先 + 偵測/占位 | 更細緻 intendedUse | 僅 `cardData.datasets` 或 `--force-dataset`；預設僅名稱，`--dataset-details` 可含 `datasetType` |
| 任務標籤來源 | 頁面啟發式 | HF API `pipeline_tag` | 同 v1.1 | 同 v1.1 | 同 v1.1 | 同 v1.1 |
| 相依套件 | 固定 `transformers` | 固定 `transformers` | `transformers`（含版本提示）+ 偵測常見依賴 | 同 v1.2 | 同 v1.2 | 僅憑證據（requirements/environment/pyproject/setup/Pipfile + README 的 pip/import）；無預設 |
| `AIPackage.packageVersion` | `main` | API `sha` 或 `main` | `sha`（釘選） | `sha`（釘選） | `sha`（釘選） | `sha`（釘選） |
| `typeOfModel`/`domain` | 固定 `Transformer`/`NLP` | 固定 | 固定 | 由 `config.json`/pipeline 推斷 | 更強啟發式 | 省略（不推斷） |
| `useSensitivePersonalInformation` | `None` | `False` | `False` | `False` | `False` | 省略 |

---

#### 小結
- v1.0 → v1.4：逐步加強「完整列檔、雜湊可得、授權更準、輸出可重現（commit 釘選）、適度推斷（型別/領域/資料集/相依）」。
- v1.5：回歸「事實為本」，僅納入可直接佐證之欄位，避免過度推斷；仍維持 commit 釘選與授權嚴謹處理，適合需要保守、可驗證輸出的情境。