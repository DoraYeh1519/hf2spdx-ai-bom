## hf2spdx-ai-bom

以 Hugging Face 倉庫為來源，產生與補強 AI-BOM（SPDX 3.0 JSON-LD）。完整技術細節請見 `src/README.md`。

### 統一入口 CLI（快速開始）

顯示說明：
```bash
python src/cli.py --help
python src/cli.py gen --help
python src/cli.py enrich hf --help
python src/cli.py enrich ailuminate --help
python src/cli.py run --help
```

產生 AI-BOM：
```bash
python src/cli.py gen <repo_id_or_url> -o output/<name>.spdx3.json --timeout 30
```

單獨補強（HF）：
```bash
python src/cli.py enrich hf <one_or_more_json> -o orig
```

單獨補強（AILuminate）：
```bash
python src/cli.py enrich ailuminate <one_or_more_json> -o orig --add-comment
```

一鍵管線：
```bash
python src/cli.py run <repo_id_or_url> --overwrite --add-comment --timeout 30
```
預設流程：gen → enrich hf → enrich ailuminate。若未指定輸出，預設為 `output/<safe_model_id>_<YYYYMMDD>.spdx3.json`。

### 設定與設計
- 可選設定檔：`configs/default.yaml`（讀不到或解析失敗會回退內建預設）。
- 不更動既有腳本；以子程序呼叫：
  - 產生：`src/hf2spdx_ai_bom.py`
  - 補強（HF）：`src/enrichers/spdx_hf_enricher.py`
  - 補強（AILuminate）：`src/enrichers/spdx_ailuminate_enricher.py`
- 旗標「有則傳、無則略」，跨平台執行使用 `sys.executable`。

### 目錄
- `src/cli.py`：統一入口 CLI
- `src/pipeline/run.py`：管線封裝（提供 `run_pipeline()`）
- `configs/default.yaml`：管線預設（可選）
- `src/README.md`：完整技術說明與歷程


