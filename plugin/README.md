## hf2spdx-ai-bom 插件（SPDX AI-BOM Enrichers）

本資料夾提供兩個「事實為本」的 SPDX 3.0 AI‑BOM 補強工具，分別從不同來源為既有的 AI‑BOM JSON 檔添加或補齊欄位：

- `spdx_ai_enricher.py`：從 Hugging Face 模型倉庫的 API/README 抽取 AI 欄位，合併回 `AIPackage`。
- `spdx_ailuminate_enricher.py`：從 MLCommons AILuminate v1.0（Bare Models）對照清單比對模型名稱，寫入 `safetyRiskAssessment`（風險等級）。

### 安裝需求
- Python 3.8+
- 套件：`requests`、`beautifulsoup4`

```bash
pip install requests beautifulsoup4
```

### 共通行為與約定
- 輸入為一或多個 SPDX 3.0 AI‑BOM JSON 檔（皆為就地更新或另存）。
- 預設為 dry‑run（只顯示將更新內容，不寫檔）；若加 `-o/--output` 才會輸出檔案。
- 日誌前綴使用 `[owner/repo]`（以 `AIPackage.name` 或 HF 連結推得；缺少時退回檔名）。
- 帶 `-o` 時，最後會列印「寫入總結」，顯示每個輸入實際寫入了哪些欄位與輸出路徑。

`-o/--output` 三態（兩個腳本皆相同）：
- `-o`（無值）：每個輸入在同目錄另存為 `enriched.<原檔名>`。
- `-o orig`（同義詞：`inplace`/`overwrite`）：就地覆寫原始檔案。
- `-o <檔名>`：單一輸入時輸出到指定檔名（多檔輸入會報錯）。

---

### 1) spdx_ai_enricher.py（HF 欄位補強）

作用：
- 從 Hugging Face API 的 `cardData` 與 README 中，抽取並補齊下列欄位至 `AIPackage`：
  - `autonomyType`
  - `domain`
  - `informationAboutApplication`
  - `informationAboutTraining`
  - `limitation`
  - `hyperparameter`（{name,value} 清單）
  - `metric`（model‑index/簡易表格）
  - `metricDecisionThreshold`

指令：
```bash
python plugin/spdx_ai_enricher.py <ai-bom.json> [更多.json ...] \
  [--timeout 30] [--dry-run] [-o [orig|inplace|overwrite|<filename>]] [--quiet]
```

輸出格式：
- 逐欄位顯示是否找到：
  - `[owner/repo] not found: <field>`
  - `[owner/repo] found <field>:` 後印出該欄位將寫入的 JSON 內容
- dry‑run 會顯示：`[owner/repo] would update fields: ...`；就地/另存會顯示「已更新欄位」與「寫入總結」。

合併策略（保留原值、避免重複）：
- `autonomyType`：字串/清單合併為去重清單（單一值仍輸出為字串），原有順序優先，新值在後。
- `domain`：合併去重為清單，保序。
- `informationAbout*`、`limitation`：原值非空則保留；原值為空時才覆寫。
- `hyperparameter`：依 `name` 合併；同名覆寫值，新名追加；保持原有→新增的順序。
- `metric`：依完整簽章去重（type/name/value/unit/dataset/split/config），追加不重複者。
- `metricDecisionThreshold`：原值非空則保留；空值時才寫入新值。

範例：
```bash
# 預覽（不寫檔）
python plugin/spdx_ai_enricher.py output_files/v2.0/v2.0.0/meta-llama_Llama-3.3-70B-Instruct.spdx3.json

# 每檔另存 enriched.<原檔名>
python plugin/spdx_ai_enricher.py output_files/v2.0/v2.0.0/*.json -o

# 就地覆寫
python plugin/spdx_ai_enricher.py output_files/v2.0/v2.0.0/meta-llama_Llama-3.3-70B-Instruct.spdx3.json -o orig
```

---

### 2) spdx_ailuminate_enricher.py（AILuminate 風險補強）

作用：
- 讀取 AILuminate v1.0 Bare Models 清單，使用保守的「名稱 tokens 子集合」比對策略，將命中的等第（excellent/very good/good/fair/poor）映射為 `safetyRiskAssessment`：
  - excellent/very good → low
  - good → medium
  - fair → high
  - poor → serious

指令：
```bash
python plugin/spdx_ailuminate_enricher.py <ai-bom.json> [更多.json ...] \
  [--timeout 30] [--dry-run] [-o [orig|inplace|overwrite|<filename>]] \
  [--force] [--add-comment]
```

選項：
- `--force`：若 `safetyRiskAssessment` 已存在，仍強制覆寫。
- `--add-comment`：在 `AIPackage.comment` 末尾加註來源（等第與來源連結）。

輸出：
- 會輸出：
  - `[owner/repo] found safetyRiskAssessment:` 與對應的 JSON 值
  - 若加 `--add-comment`，另輸出 `[owner/repo] found comment:` 的新增註記
- dry‑run 顯示 `would update fields: ...`；帶 `-o` 時附帶「寫入總結」。

範例：
```bash
# 預覽（不寫檔）
python plugin/spdx_ailuminate_enricher.py output_files/v2.0/v2.0.0/meta-llama_Llama-3.3-70B-Instruct.spdx3.json

# 就地覆寫並加註來源
python plugin/spdx_ailuminate_enricher.py output_files/v2.0/v2.0.0/*.json -o orig --add-comment
```


