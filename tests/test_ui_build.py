"""The committed Desktop plugin and dashboard bundle must be exactly what scripts/build_ui.py produces."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_ui_is_up_to_date() -> None:
    spec = importlib.util.spec_from_file_location("build_ui", ROOT / "scripts" / "build_ui.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(check=True) == 0, "run: python scripts/build_ui.py"


def test_desktop_plugin_only_imports_the_sdk_surface() -> None:
    text = (ROOT / "nightshift" / "desktop" / "plugin.js").read_text()
    imports = [
        ln.split(" from ")[-1].strip().strip("'\"") for ln in text.splitlines() if ln.startswith("import ")
    ]
    assert set(imports) <= {"@hermes/plugin-sdk", "react", "react/jsx-runtime"}
    for banned in ("eval(", "new Function", "<webview", "document.", "window."):
        assert banned not in text, banned


def test_plugin_readme_matches_repo_readme() -> None:
    spec = importlib.util.spec_from_file_location("sync_readme", ROOT / "scripts" / "sync_readme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--check"]) == 0, "run: python scripts/sync_readme.py"
    text = (ROOT / "nightshift" / "README.md").read_text()
    assert "](../docs/img/alert-chart.png)" in text
    assert "[x](https://github.com/a) [y](#safety) [z](../a.md)" in mod.render(
        "[x](https://github.com/a) [y](#safety) [z](a.md)"
    )
