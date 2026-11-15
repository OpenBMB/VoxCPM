from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("voxcpm_i18n", ROOT / "i18n.py")
i18n = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(i18n)


def test_locale_tags_resolve_to_the_catalog_language():
    assert i18n.normalize_lang("zh-CN") == "zh"
    assert i18n.normalize_lang("EN_us") == "en"
    assert i18n.normalize_lang("fr") == "en"


def test_t_uses_chinese_for_regional_chinese_tags():
    assert i18n.t("tab_train", lang="zh-Hans") == i18n.STRINGS["zh"]["tab_train"]
    assert i18n.t("tab_train", lang="en-GB") == "Training"
    assert i18n.t("missing-key", lang="zh") == "missing-key"
