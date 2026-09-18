"""Streamlit AppTest smoke for the FORESIGHT dashboard.

Two assertions that matter in CI:
  1. With real processed data, all four tabs render without exception.
  2. With an empty/absent data directory, the app shows the "no processed
     data" guard (st.error) instead of crashing.
"""

import os
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _load_source(name: str, rel_path: Path):
    """Import a script as a module under a unique name (avoids the 'app'
    package being shadowed by AppTest's script module)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_full_app_renders_all_tabs():
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception, f"AppTest raised: {at.exception}"
    assert at.error == [], "full-data run should not trigger the data guard"
    # Planning/Reorder/Stockout/Markdown each render a dataframe.
    assert len(at.dataframe) >= 4, f"expected >=4 dataframes, got {len(at.dataframe)}"


def test_theme_palettes_well_formed():
    """The app ships both dark and light token sets with usable contrast."""
    appmod = _load_source("_foresight_app", Path("app") / "app.py")
    assert set(appmod.THEMES.keys()) == {"dark", "light"}
    for base, pal in appmod.THEMES.items():
        for key in ("mode", "page_bg", "sidebar_bg", "card_bg", "text_hi",
                    "text_muted", "text_body", "border", "grid"):
            assert key in pal and pal[key], f"{base} missing {key}"
        assert pal["mode"] == base
        assert pal["text_hi"] != pal["page_bg"]          # readable foreground/background
        assert pal["text_muted"] != pal["page_bg"]
    assert appmod.get_theme_base() in ("dark", "light")
    assert appmod.current_theme() is appmod.THEMES[appmod.get_theme_base()]


def test_theme_toggle_flips_css_via_radio():
    """The in-app ☀/🌙 toggle must actually restyle the injected CSS."""
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception, f"AppTest raised: {at.exception}"
    css = next((m.value for m in at.markdown if "<style>" in m.value), "")
    assert "#0B0D13" in css, "expected dark page_bg on first paint"
    assert at.radio[0].value == "dark"

    at.radio[0].set_value("light")                      # ☀️ Light
    at.run()
    assert not at.exception, f"AppTest raised: {at.exception}"
    css = next((m.value for m in at.markdown if "<style>" in m.value), "")
    assert "#F5F6FA" in css, "expected light page_bg after ☀️ toggle"
    assert "color-scheme: light" in css

    at.radio[0].set_value("dark")                       # 🌙 Dark
    at.run()
    assert not at.exception, f"AppTest raised: {at.exception}"
    css = next((m.value for m in at.markdown if "<style>" in m.value), "")
    assert "#0B0D13" in css, "expected dark page_bg after toggling back"


def test_empty_data_guard_via_env_overwrite():
    """Point FORESIGHT_DATA_DIR at an empty dir → guard triggers, no crash."""
    with tempfile.TemporaryDirectory() as empty_dir:
        script = (
            "import sys\n"
            "from streamlit.testing.v1 import AppTest\n"
            "at = AppTest.from_file('app/app.py', default_timeout=240)\n"
            "at.run()\n"
            "assert not at.exception, at.exception\n"
            "assert len(at.dataframe) == 0, 'expected empty-data guard, got dataframes'\n"
            "assert at.error, 'expected st.error guidance for missing data'\n"
            "print('EMPTY_GUARD_OK')\n"
        )
        env = {**os.environ, "FORESIGHT_DATA_DIR": empty_dir}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"empty-data guard subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert "EMPTY_GUARD_OK" in proc.stdout