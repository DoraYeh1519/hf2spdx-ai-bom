#!/usr/bin/env python3
"""
hf2spdx-ai-bom_v1.8.2.py 的自動測試腳本
從 test.txt 檔案中的 URL 測試腳本
"""

import subprocess
import sys
import os
from pathlib import Path

def read_test_urls(filename="test.txt"):
    """從 test.txt 檔案讀取 URL"""
    try:
        with open(filename, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    except FileNotFoundError:
        print(f"錯誤：找不到 {filename}")
        return []
    except Exception as e:
        print(f"讀取 {filename} 時發生錯誤：{e}")
        return []

def test_hf2spdx_script(url, script_path="hf2spdx-ai-bom_v1.8.2.py", output_dir="./output_files/v1.8.2"):
    """使用單一 URL 測試 hf2spdx 腳本"""
    try:
        print(f"\n{'='*60}")
        print(f"測試 URL：{url}")
        print(f"{'='*60}")
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 從 URL 提取 repo_id 用於檔案命名
        if "huggingface.co/" in url:
            repo_id = url.split("huggingface.co/")[-1]
        else:
            repo_id = url
        
        # 生成輸出檔案名稱
        output_filename = f"{repo_id.replace('/', '_')}.spdx3.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # 執行 hf2spdx 腳本
        result = subprocess.run([
            sys.executable, script_path, url, "-o", output_path
        ], capture_output=True, text=True, timeout=300)  # 5 分鐘逾時
        
        if result.returncode == 0:
            print("✅ 成功")
            print("輸出：")
            print(result.stdout)
            print(f"檔案已儲存至：{output_path}")
        else:
            print("❌ 失敗")
            print("錯誤：")
            print(result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ 逾時 - 腳本執行時間過長")
        return False
    except Exception as e:
        print(f"❌ 錯誤：{e}")
        return False

def main():
    """執行自動測試的主函數"""
    print("hf2spdx-ai-bom_v1.8.2.py 自動測試")
    print("=" * 50)
    
    # 檢查主腳本是否存在
    script_path = "hf2spdx-ai-bom_v1.8.2.py"
    if not os.path.exists(script_path):
        print(f"錯誤：在當前目錄中找不到 {script_path}")
        return
    
    # 讀取測試 URL
    urls = read_test_urls()
    if not urls:
        print("在 test.txt 中找不到 URL")
        return
    
    print(f"找到 {len(urls)} 個 URL 要測試")
    
    # 測試每個 URL
    success_count = 0
    total_count = len(urls)
    
    for i, url in enumerate(urls, 1):
        print(f"\n測試 {i}/{total_count}")
        if test_hf2spdx_script(url, script_path):
            success_count += 1
    
    # 總結
    print(f"\n{'='*60}")
    print("測試總結")
    print(f"{'='*60}")
    print(f"總測試數：{total_count}")
    print(f"成功：{success_count}")
    print(f"失敗：{total_count - success_count}")
    print(f"成功率：{(success_count/total_count)*100:.1f}%")

if __name__ == "__main__":
    main()
