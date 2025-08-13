## hf2spdx-ai-bom

以 Hugging Face 模型倉庫為來源，自動產生 SPDX 3.0（JSON-LD）AI-BOM。從 HF API/頁面/檔案「可直接取得的事實」組裝出模型的可追溯性資訊（檔案、雜湊、授權、相依、資料集註記、引用等），且所有連結與雜湊皆「釘選到特定 commit」，利於重現與檢核。

### 特色
- 事實為本（facts-only）
  - 僅納入可在 HF API/頁面/檔案找到的明確資訊；避免臆測與過度推斷。
  - 檔案與套件下載連結皆釘選到 `commit`，避免主分支漂移。
- 嚴謹的檔案列表與雜湊
  - 列出 API 列舉的所有檔案；雜湊（SHA256）優先用 LFS metadata → blob 頁「Raw pointer details」→ 小檔 raw 計算。
- 授權處理
  - 若能正規化成 SPDX License ID → `LicenseExpression`；否則以 LICENSE 檔或頁面擷取文字建立 `SimpleLicensingText`。
- 相依來源（證據導向）
  - 解析 `requirements*.txt`、`environment.yml/.yaml`、`pyproject.toml`、`setup.cfg/py`、`Pipfile`。
  - 解析 README 中的安裝指令與 `import`（allowlist）。
  - v1.8.2 起：掃描倉庫內小型 `.py` 檔的 `import`（allowlist），並在匯總時做防守式過濾（排除檔名、旗標、URL 等）。
- 資料集（Dataset）
  - 僅在 API `cardData.datasets` 有資料或使用 `--force-dataset` 時加入；預設只寫名稱，開啟 `--dataset-details` 才加上簡單 `datasetType`（由 pipeline 映射）。
- 參考資訊（ExternalRef）
  - 支援 DOI；v1.7 起支援從 README 後備解析 DOI 與 arXiv。

### 安裝需求
- Python 3.8+（建議 3.9 或以上）
- 依賴：`requests`、`beautifulsoup4`

```bash
pip install requests beautifulsoup4
```

### 使用方式（v2.0.0）
- 主腳本：`hf2spdx-ai-bom_v2.0.0.py`

```bash
python hf2spdx-ai-bom_v2.0.0.py <repo_id_or_url> \
  [-o out.json] \
  [--force-dataset] \
  [--dataset-details] \
  [--timeout 30]
```

- 參數說明：
  - `repo_id_or_url`：例如 `openai-community/gpt2` 或 `https://huggingface.co/openai-community/gpt2`
  - `-o/--output`：輸出檔名。若不帶檔名，預設為 `<repo_id>.spdx3.json`；未加 `-o` 時輸出到 stdout。
  - `--force-dataset`：即使 API 無 `cardData.datasets`，也加入占位 `DatasetPackage` 與 `trainedOn`。
  - `--dataset-details`：在有 Dataset 時加入簡單 `datasetType`（由 `pipeline_tag` 映射）。
  - `--timeout`：HTTP 逾時秒數（預設 30）。

#### 範例
```bash
# 以 repo_id 產生 AI-BOM 並輸出到檔案
python hf2spdx-ai-bom_v2.0.0.py openai-community/gpt2 -o gpt2.spdx3.json

# 強制加入 Dataset，占位輸出
python hf2spdx-ai-bom_v2.0.0.py meta-llama/Llama-3.1-8B -o llama.spdx3.json --force-dataset

# 加入 Dataset 細節（datasetType）
python hf2spdx-ai-bom_v2.0.0.py openai-community/gpt2 -o gpt2.spdx3.json --dataset-details
```

### 主要輸出（SPDX 3.0 JSON-LD）
- `SpdxDocument`：`profileConformance` 包含 `core/software/simpleLicensing/ai`，視情況含 `dataset`。
- `AIPackage`：`downloadLocation` 指向 `.../tree/<commit>`；`packageVersion` 為 commit SHA。
- `File`：列出所有檔案；`downloadLocation` 指向 `.../resolve/<commit>/<path>`；可附帶 `SHA256`。
- `Package`（相依）：自證據來源蒐集（requirements/env/pyproject/setup/Pipfile、README 安裝行與 import、可選 `.py` 掃描），並以 `dependsOn` 關聯。
- 授權：模型授權以 `LicenseExpression`（SPDX）或 `SimpleLicensingText`（自 LICENSE/頁面文字），加上 `hasDeclaredLicense`/`hasConcludedLicense`。
- `ExternalRef`：DOI 與（可用時）arXiv。

### 舊版腳本
- 專案內仍保留多個舊版（`old_version/` 與部分根目錄版本），以利回顧：
  - v1.0～v1.4：逐步強化列檔/雜湊與授權、支援 commit 釘選、有限度之型別/領域/資料集/相依推斷。
  - v1.5 起：回歸「事實為本」，僅納入可直接佐證的欄位；引入 `--dataset-details` 取代 `--validate-minimal`。
  - v1.7：README 後備解析 DOI/arXiv、強化 License 偵測。
  - v1.8/1.8.1：安裝指令忽略 `-r/-c` 後的需求/限制檔，避免誤判為套件。
  - v1.8.2：新增掃描小型 `.py` 的 `import`（allowlist）與最終相依清單清理。

### 更新歷程（精簡）
- v1.0 → v1.4：
  - 從單檔到「列出全部檔案」；逐步引入 commit 釘選、雜湊補強與授權更嚴謹。
  - 局部推斷 `typeOfModel`/`domain`、Dataset 與相依（依 README/設定檔）。
- v1.5：
  - 轉為「事實為本」；引入 `--dataset-details`，移除 `--validate-minimal`。
- v1.7：
  - 新增 README 後備解析 DOI/arXiv；強化 License 偵測。
- v1.8.x：
  - 忽略安裝指令中的需求/限制檔名；v1.8.2 加入小型 `.py` 掃描與相依名單防守式清理。
 - v2.0.0：
   - 元素與關聯全面附帶 `creationInfo`；`AIPackage` 新增 `primaryPurpose`、`releaseTime`（取自 API `lastModified/createdAt`，否則回退文件建立時間）、`suppliedBy`（由 repo 擁有者推得之 `Organization`），並將供應者同步列為元素。

### 注意事項
- 請遵守 Hugging Face 之服務條款與各模型原作者授權條款。
- 產生之 AI-BOM 僅反映公開可取用之資料；若需更完整授權/訓練資料資訊，請參考原始模型卡與倉庫檔案。
- 以上 readme.md 完全由 gpt-5 生成


