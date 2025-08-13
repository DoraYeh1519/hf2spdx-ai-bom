import os
import sys
import pathlib

# Ensure src is on the path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from hf2spdx_ai_bom import normalize_spdx_license


def test_normalizes_common_variants():
    assert normalize_spdx_license('Apache 2.0') == 'Apache-2.0'
    assert normalize_spdx_license('Apache License 2.0') == 'Apache-2.0'
    assert normalize_spdx_license('MIT License') == 'MIT'
    assert normalize_spdx_license('gpl-3.0') == 'GPL-3.0-only'
    assert normalize_spdx_license('unknown') is None
