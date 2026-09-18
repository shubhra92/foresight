"""
app.py
======
Project FORESIGHT — Planning Dashboard (Modern D2C Analytics edition)
Client: NorthBay Living

A dark, 4-tab Streamlit dashboard that mirrors the "Modern D2C Analytics
Dashboard" product design while rendering real FORESIGHT data:

  • Planning Dashboard  — KPI cards, inventory risk matrix, priority actions,
                          revenue outlook (actual + forecast + 80 % CI)
  • Reorder Plan        — replenishment / purchase-order table
  • Stockout Alerts     — early-warning system with per-SKU risk bars
  • Markdown Candidates — overstock clearance recommendations

Data sources
------------
1. Local processed artefacts (data/processed) — fast, always available.
2. FastAPI scoring service (service/main.py) — live per-SKU /predict and
   /summary when the service is online; the dashboard degrades gracefully
   to local data when it is not.

Usage
-----
    streamlit run app/app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── path setup so imports work from repo root or app/ ─────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

# Imported by bare name (not app.api_client) so the module lookup never
# resolves `app` to this running script (app/app.py).
from api_client import DEFAULT_API_URL, DEFAULT_AUTH_TOKEN, ScoringService  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FORESIGHT — NorthBay Living",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Overridable in production (Docker) via FORESIGHT_DATA_DIR.
PROCESSED = Path(os.environ.get("FORESIGHT_DATA_DIR", ROOT / "data" / "processed"))

# ─────────────────────────────────────────────────────────────────────────────
# Brand palette (mirrors the D2C dashboard design)
# ─────────────────────────────────────────────────────────────────────────────

MUTED = "#6B7494"
BODY = "#A0A8C0"
ACCENT = "#F59E0B"

# ─────────────────────────────────────────────────────────────────────────────
# Theme tokens (dark default — mirrors the Figma design; light is a slate-on-
# white variant the user picks from the in-app ☀/🌙 toggle in the sidebar)
# ─────────────────────────────────────────────────────────────────────────────

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "mode":          "dark",
        "page_bg":       "#0B0D13",
        "sidebar_bg":    "#10131C",
        "card_bg":       "#13161F",
        "text_hi":       "#E8EAF0",
        "text_muted":    "#6B7494",
        "text_body":     "#A0A8C0",
        "border":        "rgba(255,255,255,0.07)",
        "border_strong": "rgba(255,255,255,0.16)",
        "grid":          "rgba(255,255,255,0.05)",
        "grid_strong":   "rgba(255,255,255,0.15)",
        "soft":          "rgba(255,255,255,0.05)",
    },
    "light": {
        "mode":          "light",
        "page_bg":       "#F5F6FA",
        "sidebar_bg":    "#FFFFFF",
        "card_bg":       "#FFFFFF",
        "text_hi":       "#0F172A",
        "text_muted":    "#64748B",
        "text_body":     "#334155",
        "border":        "rgba(15,23,42,0.10)",
        "border_strong": "rgba(15,23,42,0.18)",
        "grid":          "rgba(15,23,42,0.08)",
        "grid_strong":   "rgba(15,23,42,0.18)",
        "soft":          "rgba(15,23,42,0.05)",
    },
}

APP_THEME: dict[str, str] = THEMES["dark"]


def get_theme_base() -> str:
    """Return the configured Streamlit theme base ('dark' or 'light')."""
    try:
        base = st.get_option("theme.base")
    except Exception:
        return "dark"
    return base if base in ("dark", "light") else "dark"


def resolve_theme_base() -> str:
    """Active theme base for this session.

    The in-app ☀/🌙 toggle (session_state['app_theme']) wins; falls back to
    the configured Streamlit base so the first paint honours config.toml.
    """
    try:
        base = st.session_state.get("app_theme")
    except Exception:
        base = None
    if base not in ("dark", "light"):
        base = get_theme_base()
    return base


def init_theme_from_url() -> None:
    """Honour a ?theme=dark|light query param on first load (persists the
    sunny/moon toggle across browser reloads)."""
    if "app_theme" in st.session_state:
        return
    qp = st.query_params.get("theme")
    if isinstance(qp, list):
        qp = qp[0] if qp else None
    if qp in ("dark", "light"):
        st.session_state["app_theme"] = qp


def sync_theme_to_url() -> None:
    """Mirror the active theme into the URL so reloads keep the choice."""
    base = st.session_state.get("app_theme")
    if base in ("dark", "light"):
        st.query_params["theme"] = base


def current_theme() -> dict[str, str]:
    return THEMES[resolve_theme_base()]

CAT_COLORS = {
    "Furniture": "#F59E0B",
    "Decor": "#A78BFA",
    "Lighting": "#38BDF8",
    "Kitchen": "#FB923C",
    "Bedding": "#34D399",
}

ALERT_CONFIG = {
    "Critical": {"bg": "rgba(248,113,113,0.15)", "text": "#F87171", "border": "rgba(248,113,113,0.4)"},
    "High": {"bg": "rgba(251,146,60,0.15)", "text": "#FB923C", "border": "rgba(251,146,60,0.4)"},
    "Medium": {"bg": "rgba(245,158,11,0.15)", "text": "#F59E0B", "border": "rgba(245,158,11,0.4)"},
    "Low": {"bg": "rgba(107,116,148,0.15)", "text": "#8892B0", "border": "rgba(107,116,148,0.3)"},
}

STATUS_CONFIG = {
    "Raise Now": {"bg": "rgba(248,113,113,0.15)", "text": "#F87171", "border": "rgba(248,113,113,0.4)"},
    "Scheduled": {"bg": "rgba(52,211,153,0.12)", "text": "#34D399", "border": "rgba(52,211,153,0.3)"},
    "Pending": {"bg": "rgba(245,158,11,0.15)", "text": "#F59E0B", "border": "rgba(245,158,11,0.4)"},
}

MD_ACTION_CONFIG = {
    "Markdown": {"bg": "rgba(248,113,113,0.12)", "text": "#F87171", "border": "rgba(248,113,113,0.35)"},
    "Flash Sale": {"bg": "rgba(251,146,60,0.12)", "text": "#FB923C", "border": "rgba(251,146,60,0.35)"},
    "Bundle": {"bg": "rgba(167,139,250,0.12)", "text": "#A78BFA", "border": "rgba(167,139,250,0.35)"},
    "Monitor": {"bg": "rgba(107,116,148,0.12)", "text": "#8892B0", "border": "rgba(107,116,148,0.3)"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_weekly() -> pd.DataFrame:
    p = PROCESSED / "weekly_features.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(PROCESSED / "weekly_features.csv")
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["category"] = df["category"].astype(str).str.strip().str.title()
    return df


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    p = PROCESSED / "forecast.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(PROCESSED / "forecast.csv")
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["category"] = df["category"].astype(str).str.strip().str.title()
    return df


@st.cache_data(show_spinner=False)
def load_risk() -> pd.DataFrame:
    p = PROCESSED / "risk_scores.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(PROCESSED / "risk_scores.csv")
    df["category"] = df["category"].astype(str).str.strip().str.title()
    return df


@st.cache_data(show_spinner=False)
def load_eval() -> Optional[dict]:
    """Rolling-origin backtest evidence persisted by src/forecast.py."""
    p = ROOT / "reports" / "backtest_metrics.json"
    if not p.exists():
        return None
    import json
    return json.load(open(p))


@st.cache_data(show_spinner=False)
def load_accuracy_history() -> pd.DataFrame:
    """Model-vs-baseline WAPE snapshots appended by src/run_weekly.py."""
    p = ROOT / "reports" / "accuracy_tracking.parquet"
    if not p.exists():
        return pd.DataFrame()
    hist = pd.read_parquet(p)
    for col in ("run_at", "forecast_start", "forecast_end", "data_end_date"):
        if col in hist.columns:
            hist[col] = pd.to_datetime(hist[col], errors="coerce")
        else:
            hist[col] = pd.NaT
    for col in ("wape_lgbm", "wape_baseline", "improvement_pct"):
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist


def _check_data() -> bool:
    return all(
        (PROCESSED / f).exists()
        for f in ["weekly_features.parquet", "forecast.parquet", "risk_scores.parquet"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers (Indian numbering)
# ─────────────────────────────────────────────────────────────────────────────

def inr_group(n: float) -> str:
    """'₹2,14,83,600' — exact Indian digit grouping."""
    if pd.isna(n):
        return "—"
    n = int(round(float(n)))
    sign = "-" if n < 0 else ""
    n = abs(n)
    s = str(n)
    if len(s) <= 3:
        return f"{sign}₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    groups = []
    while len(rest) > 0:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    return f"{sign}₹{','.join(groups)},{last3}"


def inr_short(n: float) -> str:
    if pd.isna(n):
        return "—"
    n = float(n)
    if abs(n) >= 1e7:
        return f"₹{n / 1e7:.2f}Cr"
    if abs(n) >= 1e5:
        return f"₹{n / 1e5:.1f}L"
    if abs(n) >= 1e3:
        return f"₹{n / 1e3:.0f}K"
    return f"₹{n:,.0f}"


def pct_str(p: float) -> str:
    return f"{'+' if p >= 0 else ''}{p:.1f}%"


def days_str(days: float) -> str:
    return f"{int(round(days))}d"


# ─────────────────────────────────────────────────────────────────────────────
# CSS + HTML primitives
# ─────────────────────────────────────────────────────────────────────────────

def inject_css(theme: Optional[dict] = None) -> None:
    t = theme or APP_THEME
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Instrument+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{ font-family: 'Instrument Sans', sans-serif; }}
        .stApp {{ background: {t['page_bg']}; color: {t['text_hi']}; -webkit-font-smoothing: antialiased; }}
        section[data-testid="stSidebar"] {{ background: {t['sidebar_bg']}; border-right: 1px solid {t['border']}; }}
        .fs-label {{
            font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.12em;
            text-transform: uppercase; color: {t['text_muted']};
        }}
        .fs-sub {{ font-family: 'Instrument Sans', sans-serif; font-size: 12px; color: {t['text_muted']}; }}
        .fs-kpi {{
            background: {t['card_bg']}; border: 1px solid {t['border']}; border-radius: 4px;
            padding: 22px 24px; display: flex; flex-direction: column; gap: 12px; height: 100%;
        }}
        .fs-kpi-value {{
            font-family: 'JetBrains Mono', monospace; font-size: 25px; font-weight: 600;
            color: {t['text_hi']}; line-height: 1.05; display: flex; align-items: flex-end;
            justify-content: space-between; gap: 8px;
        }}
        .fs-delta {{
            font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500;
            padding: 2px 8px; border-radius: 3px; white-space: nowrap;
        }}
        .fs-underline {{ height: 1px; }}
        .fs-section-eyebrow {{
            font-family: 'Instrument Sans', sans-serif; font-size: 11px; letter-spacing: 0.12em;
            text-transform: uppercase; color: {t['text_muted']}; margin-bottom: 4px;
        }}
        .fs-section-title {{
            font-family: 'Instrument Sans', sans-serif; font-size: 14px; font-weight: 500; color: {t['text_hi']};
        }}
        .stTabs [data-baseweb="tab"] {{ font-family: 'Instrument Sans', sans-serif; font-size: 13px; }}

        /* ── native chrome forced to the active palette (both modes) ── */
        :root, .stApp, section[data-testid="stSidebar"] {{ color-scheme: {t['mode']}; }}
        * {{ scrollbar-color: {t['border_strong']} transparent; }}
        [data-testid="stMarkdownContainer"] {{ color: {t['text_body']}; }}
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6 {{ color: {t['text_hi']}; }}
        [data-testid="stMarkdownContainer"] hr {{ border-color: {t['border_strong']}; }}
        [data-testid="stWidgetLabel"], .stTextInput label, .stNumberInput label,
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {t['text_muted']}; }}
        .stTextInput input, .stNumberInput input, .stDateInput input, textarea,
        [data-testid="stTextInput"] input {{ background: {t['card_bg']}; color: {t['text_hi']}; caret-color: {ACCENT}; }}
        .stTextInput div[data-baseweb="input"], .stNumberInput div[data-baseweb="input"] {{ border-color: {t['border_strong']}; }}
        .stTextInput input::placeholder, .stNumberInput input::placeholder {{ color: {t['text_muted']}; }}
        [data-baseweb="select"] > div {{ background: {t['card_bg']} !important; border-color: {t['border_strong']} !important; }}
        [data-baseweb="select"] [data-testid="stMarkdownContainer"] {{ color: {t['text_hi']}; }}
        [data-baseweb="tag"] {{ background: {t['soft']}; border-color: {t['border_strong']}; }}
        [data-baseweb="tag"] [data-testid="stMarkdownContainer"] {{ color: {t['text_hi']}; }}
        [data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="menu-list"] {{ background: {t['sidebar_bg']}; }}
        [data-baseweb="menu"] li, [data-baseweb="menu"] li * {{ color: {t['text_body']}; }}
        [data-baseweb="menu"] li:hover {{ background: {t['soft']}; }}
        .stButton button {{ background: {t['soft']}; color: {t['text_hi']}; border: 1px solid {t['border_strong']}; }}
        .stButton button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
        .stTabs [data-baseweb="tab-list"] {{ border-bottom-color: {t['border']}; }}
        .stTabs [data-baseweb="tab"] {{ color: {t['text_muted']}; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{ color: {t['text_hi']}; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT}; }}
        [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {{ color: {t['text_body']}; }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{ color-scheme: {t['mode']}; }}
        [data-testid="stExpander"] {{ border-color: {t['border_strong']}; }}
        [data-testid="stExpander"] summary {{ color: {t['text_hi']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card_html(label: str, value: str, delta: Optional[str], good: Optional[bool],
                  sub: str, accent: str) -> str:
    if delta is not None and good is not None:
        color = "#34D399" if good else "#F87171"
        bg = "rgba(52,211,153,0.1)" if good else "rgba(248,113,113,0.1)"
        bd = "rgba(52,211,153,0.25)" if good else "rgba(248,113,113,0.25)"
        pill = f'<span class="fs-delta" style="color:{color};background:{bg};border:1px solid {bd}">{delta}</span>'
    else:
        pill = '<span class="fs-delta" style="color:#6B7494;background:rgba(107,116,148,0.08);border:1px solid rgba(107,116,148,0.2)">—</span>'
    return (
        f'<div class="fs-kpi">'
        f'<span class="fs-label">{label}</span>'
        f'<div class="fs-kpi-value">{value}{pill}</div>'
        f'<div class="fs-underline" style="background:linear-gradient(90deg, {accent}40, transparent)"></div>'
        f'<span class="fs-sub">{sub}</span>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# API service helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_service() -> ScoringService:
    base = st.session_state.get("api_url", "") or DEFAULT_API_URL
    token = st.session_state.get("api_token", "") or DEFAULT_AUTH_TOKEN
    return ScoringService(base_url=base, token=token)


def check_api() -> ScoringService:
    svc = get_service()
    ok = svc.check_health()
    st.session_state["api_ok"] = ok
    st.session_state["api_status"] = svc.status_text()
    st.session_state["api_checked"] = True
    return svc

# ─────────────────────────────────────────────────────────────────────────────
# KPI computation (real data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(weekly: pd.DataFrame, risk: pd.DataFrame) -> list[dict]:
    weekly = weekly.copy()
    last_week = weekly["week_start"].max()
    day = pd.Timedelta(days=7)

    rev_last4 = weekly[weekly["week_start"] > last_week - 4 * day]["revenue"].sum()
    rev_prev4 = weekly[
        (weekly["week_start"] <= last_week - 4 * day)
        & (weekly["week_start"] > last_week - 8 * day)
    ]["revenue"].sum()
    rev_delta = (rev_last4 - rev_prev4) / rev_prev4 * 100 if rev_prev4 else 0.0

    inv_now = 0.0
    inv_prev = 0.0
    weeks_sorted = sorted(weekly["week_start"].unique())
    if len(weeks_sorted) >= 1:
        mask = weekly["week_start"] == weeks_sorted[-1]
        inv_now = float((weekly.loc[mask, "on_hand_units"] * weekly.loc[mask, "unit_cost"]).sum())
    if len(weeks_sorted) >= 2:
        mask = weekly["week_start"] == weeks_sorted[-2]
        inv_prev = float((weekly.loc[mask, "on_hand_units"] * weekly.loc[mask, "unit_cost"]).sum())
    inv_delta = (inv_now - inv_prev) / inv_prev * 100 if inv_prev else 0.0

    reorder = risk[risk["quadrant"] == "REORDER NOW"]
    markdown = risk[risk["quadrant"] == "MARKDOWN / CLEAR"]

    sales_at_risk = float(reorder["sales_at_risk_inr"].sum())
    critical = int((reorder["stockout_risk"] >= 0.8).sum())
    capital_locked = float(markdown["locked_capital_inr"].sum())
    excess = int(markdown["overstock_risk"].apply(lambda r: int(r >= 0.6)).sum()) or len(markdown)

    return [
        {
            "label": "Gross Revenue",
            "value": inr_group(rev_last4),
            "delta": pct_str(rev_delta),
            "good": rev_delta >= 0,
            "sub": "vs prev 30 days",
            "accent": "#F59E0B",
        },
        {
            "label": "Inventory Value",
            "value": inr_group(inv_now),
            "delta": pct_str(inv_delta),
            "good": inv_delta >= 0,
            "sub": f"active SKUs: {int(weekly['sku_id'].nunique())}",
            "accent": "#38BDF8",
        },
        {
            "label": "Sales at Risk \u2014 Stockout",
            "value": inr_group(sales_at_risk),
            "delta": None,
            "good": None,
            "sub": f"{critical} SKUs critical",
            "accent": "#F87171",
        },
        {
            "label": "Capital Locked \u2014 Overstock",
            "value": inr_group(capital_locked),
            "delta": None,
            "good": None,
            "sub": f"{excess} SKUs excess stock",
            "accent": "#A78BFA",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Plot style helper
# ─────────────────────────────────────────────────────────────────────────────

def base_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    t = APP_THEME
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", size=10, color=t["text_muted"]),
        margin=dict(l=8, r=16, t=16, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False, tickfont=dict(color=t["text_muted"]))
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False, tickfont=dict(color=t["text_muted"]))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Risk matrix scatter (stockout vs overstock)
# ─────────────────────────────────────────────────────────────────────────────

def render_risk_matrix(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No SKUs match the current filters.")
        return

    df = df.copy()
    max_stake = max(float(df["rupee_at_stake"].max()), 1.0)
    df["bubble"] = (df["rupee_at_stake"] / max_stake * 34).clip(lower=6)

    fig = go.Figure()
    for cat, color in CAT_COLORS.items():
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["overstock_risk"] * 100,
            y=sub["stockout_risk"] * 100,
            mode="markers",
            name=cat,
            marker=dict(
                size=sub["bubble"],
                color=color,
                opacity=0.85,
                line=dict(width=0.6, color=APP_THEME["border_strong"]),
            ),
            customdata=sub[["sku_id", "subcategory", "rupee_at_stake"]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                "Stockout %{y:.1f}% \u00b7 Overstock %{x:.1f}%<br>"
                "\u20b9 at stake %{customdata[2]:,.0f}<extra></extra>"
            ),
        ))

    fig.add_hline(y=50, line_dash="dot", line_color=APP_THEME["grid_strong"])
    fig.add_vline(x=50, line_dash="dot", line_color=APP_THEME["grid_strong"])

    fig.add_annotation(x=46, y=96, text="HIGH STOCKOUT", showarrow=False,
                       font=dict(size=9, color="rgba(248,113,113,0.6)"))
    fig.add_annotation(x=96, y=96, text="DUAL RISK", showarrow=False,
                       font=dict(size=9, color="rgba(248,113,113,0.45)"))
    fig.add_annotation(x=46, y=4, text="HEALTHY", showarrow=False,
                       font=dict(size=9, color="rgba(107,116,148,0.6)"))
    fig.add_annotation(x=96, y=4, text="HIGH OVERSTOCK", showarrow=False,
                       font=dict(size=9, color="rgba(167,139,250,0.6)"))

    fig.update_xaxes(title=dict(text="Overstock Risk %"), range=[-4, 104])
    fig.update_yaxes(title=dict(text="Stockout Risk %"), range=[-4, 106])
    st.plotly_chart(base_layout(fig, height=320), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Revenue outlook (actual + forecast + 80 % CI)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def portfolio_series(weekly: pd.DataFrame, forecast: pd.DataFrame, history_weeks: int = 30):
    weekly = weekly.copy()
    last_week = weekly["week_start"].max()

    actual = (
        weekly[weekly["week_start"] > last_week - pd.Timedelta(weeks=history_weeks)]
        .groupby("week_start")["revenue"]
        .sum()
        .reset_index()
    )
    actual["kind"] = "actual"
    actual.columns = ["week_start", "value", "kind"]

    fc = forecast.copy()
    fc["rev"] = fc["forecast"] * fc["list_price"]
    fc["lo"] = fc["forecast_lo"] * fc["list_price"]
    fc["hi"] = fc["forecast_hi"] * fc["list_price"]
    group = (
        fc.sort_values("week_start")
        .groupby("week_start")
        .agg(value=("rev", "sum"), lo=("lo", "sum"), hi=("hi", "sum"))
        .reset_index()
    )
    group["kind"] = "forecast"

    merged = pd.concat(
        [actual.assign(lo=np.nan, hi=np.nan), group],
        ignore_index=True,
        sort=False,
    )
    merged["label"] = merged["week_start"].dt.strftime("%b %y")
    return merged


def render_revenue_outlook(weekly: pd.DataFrame, forecast: pd.DataFrame) -> None:
    data = portfolio_series(weekly, forecast)
    if data.empty:
        st.info("No forecast data available.")
        return

    fig = go.Figure()

    band = data[data["kind"] == "forecast"]
    if not band.empty:
        xs = pd.concat([band["week_start"], band["week_start"][::-1]])
        ys = pd.concat([band["hi"], band["lo"][::-1]])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", mode="none",
            fillcolor="rgba(245,158,11,0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% CI", hoverinfo="skip",
        ))

    actual = data[data["kind"] == "actual"]
    fig.add_trace(go.Scatter(
        x=actual["week_start"], y=actual["value"], mode="lines+markers",
        name="Actual", line=dict(color=APP_THEME["text_hi"], width=2),
        marker=dict(size=3, color=APP_THEME["text_hi"]),
    ))

    fc_line = data[data["kind"] == "forecast"]
    fig.add_trace(go.Scatter(
        x=fc_line["week_start"], y=fc_line["value"], mode="lines+markers",
        name="Forecast", line=dict(color="#F59E0B", width=2, dash="6 3"),
        marker=dict(size=4, color="#F59E0B"),
    ))

    if not actual.empty and not fc_line.empty:
        x_today = float(pd.Timestamp(actual["week_start"].max()).timestamp() * 1000)
        fig.add_vline(
            x=x_today, line_dash="4 4",
            line_color="rgba(245,158,11,0.45)",
            annotation_text="TODAY", annotation_position="top",
            annotation_font=dict(color="#F59E0B", size=9),
        )

    fig.update_xaxes(tickformat="%b %y", dtick="M1")
    fig.update_yaxes(tickformat=".2s", tickprefix="\u20b9")
    st.plotly_chart(base_layout(fig, height=300), use_container_width=True)

    st.markdown(
        f'<div style="text-align:right;font-family:JetBrains Mono,monospace;font-size:10px;'
        f'color:{APP_THEME["text_muted"]};letter-spacing:0.06em">Model: FORESIGHT v1 \u00b7 LightGBM (WAPE 0.12 vs baseline 0.26) \u00b7 '
        f'Horizon {int(fc_line["week_start"].nunique())} weeks \u00b7 Trained on 24mo sales history \u00b7 '
        f'Backtested rolling-origin CV</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-SKU forecast drill-down (live API when online, else cached)
# ─────────────────────────────────────────────────────────────────────────────

def render_sku_forecast(sku_id: str, weekly: pd.DataFrame, forecast: pd.DataFrame,
                        live: Optional[dict] = None, live_ok: bool = False) -> None:
    hist = (
        weekly[weekly["sku_id"] == sku_id]
        .sort_values("week_start")
        .tail(20)
        .reset_index(drop=True)
    )
    if hist.empty:
        st.info(f"No historical data for **{sku_id}**.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["week_start"], y=hist["units_sold"], mode="lines+markers",
        name="Actual", line=dict(color=APP_THEME["text_hi"], width=2), marker=dict(size=4),
    ))

    if live_ok and live and live.get("results"):
        res = live["results"][0]
        wks = [fw["week_start"] for fw in res.get("forecast", [])]
        vals = [fw["forecast"] for fw in res.get("forecast", [])]
        lo = [fw.get("forecast_lo", v * 0.8) for v, fw in zip(vals, res.get("forecast", []))]
        hi = [fw.get("forecast_hi", v * 1.2) for v, fw in zip(vals, res.get("forecast", []))]
        fig.add_trace(go.Scatter(
            x=wks + wks[::-1], y=hi + lo[::-1], fill="toself", mode="none",
            fillcolor="rgba(245,158,11,0.14)",
            line=dict(color="rgba(255,255,255,0)"), name="80% CI",
        ))
        fig.add_trace(go.Scatter(
            x=wks, y=vals, mode="lines+markers", name="Forecast (live)",
            line=dict(color="#F59E0B", width=2, dash="6 3"), marker=dict(size=4),
        ))
        caption = "Source: **FastAPI \u00b7 POST /predict (live)**"
    else:
        fc = forecast[forecast["sku_id"] == sku_id].sort_values("week_start")
        if not fc.empty:
            xs = pd.concat([fc["week_start"], fc["week_start"][::-1]])
            ys = pd.concat([fc["forecast_hi"], fc["forecast_lo"][::-1]])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, fill="toself", mode="none",
                fillcolor="rgba(245,158,11,0.14)",
                line=dict(color="rgba(255,255,255,0)"), name="80% CI",
            ))
            fig.add_trace(go.Scatter(
                x=fc["week_start"], y=fc["forecast"], mode="lines+markers",
                name="Forecast (cached)",
                line=dict(color="#F59E0B", width=2, dash="6 3"), marker=dict(size=4),
            ))
        caption = "Source: **local forecast cache** \u00b7 Live API offline"

    fig.update_xaxes(tickformat="%b %y", dtick="M1")
    fig.update_yaxes(title=dict(text="Units / Week"))
    st.plotly_chart(base_layout(fig, height=250), use_container_width=True)
    st.caption(caption)

# ─────────────────────────────────────────────────────────────────────────────
# Styled table helper (pill columns via pandas Styler)
# ─────────────────────────────────────────────────────────────────────────────

def _pill_css(val, cfg: dict) -> str:
    c = cfg.get(str(val), cfg.get("default", {"bg": "rgba(0,0,0,0)", "text": APP_THEME["text_muted"], "border": APP_THEME["border"]}))
    return (
        f"background-color:{c['bg']}; color:{c['text']};"
        f"font-family:'JetBrains Mono',monospace; font-size:11px;"
        f"border:1px solid {c['border']}; border-radius:3px; text-align:center;"
    )


def render_table(df: pd.DataFrame, pill_cols: Optional[dict] = None,
                 mono_cols: Optional[list[str]] = None) -> pd.io.formats.style.Styler:
    df = df.reset_index(drop=True)
    styler = df.style

    for col in mono_cols or []:
        if col in df.columns:
            styler = styler.map(
                lambda v: "font-family:'JetBrains Mono',monospace;",
                subset=[col],
            )

    for col, cfg in (pill_cols or {}).items():
        if col in df.columns:
            styler = styler.map(lambda v: _pill_css(v, cfg), subset=[col])

    styler = styler.set_properties(
        **{
            "font-family": "'Instrument Sans', sans-serif",
            "font-size": "12px",
            "background-color": "rgba(0,0,0,0)",
            "color": APP_THEME["text_body"],
        }
    )
    return styler


def download_csv(df: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(label=label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Demand trend helper
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def demand_trend_map(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku, grp in weekly.sort_values("week_start").groupby("sku_id"):
        recent = grp["units_sold"].tail(4).sum()
        prior = grp["units_sold"].iloc[-8:-4].sum()
        pct = ((recent - prior) / prior * 100) if prior else 0.0
        rows.append({"sku_id": sku, "trend_pct": float(pct)})
    return pd.DataFrame(rows)


def urgency_of(score: float) -> str:
    if score >= 0.9:
        return "Critical"
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Planning Dashboard
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Forecast accuracy panel — rolling-origin backtest evidence (WAPE vs baseline)
# ─────────────────────────────────────────────────────────────────────────────

def render_accuracy_panel(evals: Optional[dict]) -> None:
    if not evals:
        st.info(
            "No backtest evidence found. Run `python src/forecast.py` to generate "
            "`reports/backtest_metrics.json`.",
            icon="\u2139\ufe0f",
        )
        return

    wc = evals.get("wape_comparison", {})
    lgbm = evals.get("lgbm", {})
    base = evals.get("baseline", {})

    st.markdown(
        '<div class="fs-section-eyebrow">Backtest Evidence \u2014 Walk-Forward</div>'
        '<div class="fs-section-title">Forecast Accuracy \u2014 WAPE vs Seasonal-Naive Baseline</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        kpi_card_html("Model WAPE", f"{wc.get('lgbm_wape', 0) * 100:.1f}%",
                      None, None, f"LightGBM \u00b7 {str(evals.get('model_type', 'lgbm'))}", ACCENT),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi_card_html("Baseline WAPE", f"{wc.get('baseline_wape', 0) * 100:.1f}%",
                      None, None, "seasonal-naive \u00b7 same week last yr", MUTED),
        unsafe_allow_html=True,
    )
    imp = wc.get("improvement_pct")
    imp_val = f"{imp:.0f}% better" if imp is not None else "—"
    c3.markdown(
        kpi_card_html("Improvement", imp_val, None, None,
                      "lower is better \u00b7 WAPE = \u03a3|a\u2212p| / \u03a3|a|",
                      "#34D399" if (imp or 0) > 0 else "#F87171"),
        unsafe_allow_html=True,
    )

    folds = (lgbm.get("folds") or []) if not isinstance(lgbm.get("folds"), dict) else []
    if folds:
        fold_df = pd.DataFrame(folds)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fold_df["fold"].astype(str),
            y=fold_df["wape"] * 100,
            marker_color="#F59E0B",
            hovertemplate="Fold %{x}: WAPE %{y:.1f}%<extra></extra>",
        ))
        base_wape = (base.get("wape") or wc.get("baseline_wape"))
        if base_wape:
            fig.add_hline(
                y=float(base_wape) * 100,
                line_dash="4 4", line_color="rgba(160,168,192,0.6)",
                annotation_text=f"Baseline {float(base_wape) * 100:.1f}%",
                annotation_position="top right",
                annotation_font=dict(color="#8892B0", size=9),
            )
        fig.update_xaxes(title=dict(text="Rolling-origin fold"))
        fig.update_yaxes(title=dict(text="WAPE %"))
        st.plotly_chart(
            base_layout(fig, height=240),
            use_container_width=True,
        )
        st.caption(
            f"{evals.get('cv', {}).get('method', 'rolling-origin CV')} \u2014 "
            f"{evals.get('cv', {}).get('n_splits', '')} folds "
            f"\u00b7 {evals.get('cv', {}).get('test_size_weeks', '')}-week test windows "
            f"\u00b7 no future data in any feature \u00b7 generated {str(evals.get('generated_at', ''))[:10]}"
        )

    # Drift monitoring — WAPE across weekly refreshes (src/run_weekly.py)
    hist = load_accuracy_history()
    if not hist.empty and {"run_at", "wape_lgbm", "wape_baseline"}.issubset(hist.columns):
        hist = hist.sort_values("run_at").dropna(subset=["run_at"])
        if len(hist) >= 1:
            last = hist.iloc[-1]
            st.markdown(
                '<div class="fs-section-eyebrow">Drift Monitoring</div>'
                '<div class="fs-section-title">Forecast Accuracy Over Time</div>',
                unsafe_allow_html=True,
            )
            gh = go.Figure()
            gh.add_trace(go.Scatter(
                x=hist["run_at"], y=hist["wape_lgbm"] * 100,
                name="LightGBM", mode="lines+markers",
                line=dict(color="#F59E0B", width=2),
                marker=dict(size=6),
                hovertemplate="%{x|%Y-%m-%d}: WAPE %{y:.1f}%<extra>LightGBM</extra>",
            ))
            gh.add_trace(go.Scatter(
                x=hist["run_at"], y=hist["wape_baseline"] * 100,
                name="Baseline", mode="lines+markers",
                line=dict(color="#8892B0", width=2, dash="4 4"),
                marker=dict(size=5),
                hovertemplate="%{x|%Y-%m-%d}: WAPE %{y:.1f}%<extra>Baseline</extra>",
            ))
            gh.update_xaxes(title=dict(text="Refresh date"))
            gh.update_yaxes(title=dict(text="WAPE %"), range=[0, None])
            st.plotly_chart(base_layout(gh, height=180), use_container_width=True)
            st.caption(
                f"Latest snapshot: {pd.to_datetime(last['run_at']):%Y-%m-%d} \u00b7 "
                f"model WAPE {float(last['wape_lgbm']) * 100:.1f}% vs "
                f"baseline {float(last['wape_baseline']) * 100:.1f}% \u00b7 "
                f"data through {pd.to_datetime(last['data_end_date']):%Y-%m-%d}"
            )


def page_dashboard(weekly: pd.DataFrame, forecast: pd.DataFrame,
                   risk: pd.DataFrame, filtered: pd.DataFrame) -> None:
    kpis = compute_kpis(weekly, risk)

    cols = st.columns(4)
    for col, kpi in zip(cols, kpis):
        col.markdown(
            kpi_card_html(kpi["label"], kpi["value"], kpi["delta"], kpi["good"], kpi["sub"], kpi["accent"]),
            unsafe_allow_html=True,
        )

    render_accuracy_panel(load_eval())

    left, right = st.columns([2, 3], gap="large")

    with left:
        st.markdown(
            '<div class="fs-section-eyebrow">Inventory Risk Matrix</div>'
            '<div class="fs-section-title">Stockout vs Overstock Risk</div>',
            unsafe_allow_html=True,
        )
        render_risk_matrix(filtered)

    with right:
        st.markdown(
            '<div class="fs-section-eyebrow">Priority Actions</div>'
            '<div class="fs-section-title">Items requiring attention</div>',
            unsafe_allow_html=True,
        )
        reorder = (
            filtered[filtered["quadrant"] == "REORDER NOW"]
            .sort_values("rupee_at_stake", ascending=False)
            .head(10)
            .copy()
        )
        if reorder.empty:
            st.success("No SKUs currently at high stockout risk. \U0001f389")
        else:
            reorder["urgency"] = reorder["stockout_risk"].map(urgency_of)
            reorder["Action Required"] = reorder.apply(
                lambda r: f"{r['recommended_action']} \u2014 {days_str(r['stockout_cover_weeks'] * 7)} cover left",
                axis=1,
            )
            tbl = pd.DataFrame({
                "SKU": reorder["sku_id"],
                "Item": reorder["subcategory"],
                "Category": reorder["category"],
                "Action Required": reorder["Action Required"],
                "Impact": reorder["sales_at_risk_inr"].map(inr_short),
                "Urgency": reorder["urgency"],
            })
            st.dataframe(
                render_table(tbl, pill_cols={"Urgency": ALERT_CONFIG}, mono_cols=["SKU", "Impact"]),
                use_container_width=True,
                hide_index=True,
                height=360,
            )
            download_csv(reorder[["sku_id", "category", "subcategory", "recommended_action",
                                  "sales_at_risk_inr", "stockout_risk"]],
                         "priority_actions.csv", "\u2b07 Export Actions")

    st.markdown(
        '<div class="fs-section-eyebrow">Actual vs Forecast</div>'
        '<div class="fs-section-title">Revenue Outlook \u2014 portfolio weekly revenue with 80 % confidence interval</div>',
        unsafe_allow_html=True,
    )

    sku_scope = st.selectbox(
        "SKU drill-down (live via scoring API when online)",
        options=["\u2014 portfolio \u2014"] + sorted(weekly["sku_id"].unique().tolist()),
        key="sku_scope",
    )

    live: Optional[dict] = None
    live_ok = False
    if sku_scope != "\u2014 portfolio \u2014":
        svc = get_service()
        if svc.check_health():
            try:
                live = svc.predict([sku_scope], horizon_weeks=8)
                live_ok = bool(live.get("results"))
            except Exception:
                live_ok = False
        render_sku_forecast(sku_scope, weekly, forecast, live=live, live_ok=live_ok)
    else:
        render_revenue_outlook(weekly, forecast)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Reorder Plan
# ─────────────────────────────────────────────────────────────────────────────

def page_reorder(filtered: pd.DataFrame) -> None:
    reorder = filtered[filtered["quadrant"] == "REORDER NOW"].copy()
    reorder = reorder.sort_values("rupee_at_stake", ascending=False)

    if reorder.empty:
        st.success("No SKUs currently at high stockout risk. \U0001f389")
        return

    reorder["po_value"] = reorder["suggested_reorder_qty"] * reorder["unit_cost"]
    reorder["days_left"] = reorder["stockout_cover_weeks"] * 7
    reorder["status"] = reorder["stockout_risk"].apply(lambda r: "Raise Now" if r >= 0.8 else "Scheduled")

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card_html("Total PO Value Required", inr_group(reorder["po_value"].sum()),
                              None, None, f"{len(reorder)} purchase orders", ACCENT), unsafe_allow_html=True)
    c2.markdown(kpi_card_html("Raise Immediately", str(int((reorder["status"] == "Raise Now").sum())),
                              None, None, "SKUs \u2014 critical cover", "#F87171"), unsafe_allow_html=True)
    c3.markdown(kpi_card_html("Avg Supplier Lead Time", f"{reorder['lead_time_days'].mean():.0f} days",
                              None, None, "across active vendors", "#38BDF8"), unsafe_allow_html=True)

    st.markdown(
        f'<div class="fs-section-eyebrow">Purchase Orders</div>'
        f'<div class="fs-section-title">Reorder Plan \u2014 week of {datetime.now().strftime("%b %d, %Y")}</div>',
        unsafe_allow_html=True,
    )

    tbl = pd.DataFrame({
        "SKU": reorder["sku_id"],
        "Item": reorder["subcategory"],
        "Category": reorder["category"],
        "Current Stock": reorder["on_hand_units"].astype(str) + " units",
        "Days Left": reorder["days_left"].round(0).astype(int).astype(str) + "d",
        "Reorder Qty": reorder["suggested_reorder_qty"].astype(str) + " units",
        "Unit Cost": reorder["unit_cost"].map(inr_group),
        "PO Value": reorder["po_value"].map(inr_group),
        "Lead Time": reorder["lead_time_days"].astype(str) + "d",
        "Status": reorder["status"],
    })
    st.dataframe(
        render_table(
            tbl,
            pill_cols={"Status": STATUS_CONFIG},
            mono_cols=["SKU", "Days Left", "Reorder Qty", "Unit Cost", "PO Value", "Lead Time"],
        ),
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    download_csv(reorder[["sku_id", "category", "subcategory", "on_hand_units",
                          "suggested_reorder_qty", "unit_cost", "po_value", "lead_time_days"]],
                 "reorder_plan.csv", "\u2b07 Export Plan")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Stockout Alerts
# ─────────────────────────────────────────────────────────────────────────────

def page_stockout(weekly: pd.DataFrame, filtered: pd.DataFrame) -> None:
    alert = filtered[filtered["stockout_risk"] >= 0.5].copy()
    if alert.empty:
        st.success("No stockout risk SKUs. \U0001f389")
        return

    trends = demand_trend_map(weekly)
    alert = alert.merge(trends, on="sku_id", how="left")
    alert["days_cover"] = alert["stockout_cover_weeks"] * 7
    alert["alert_pill"] = alert["stockout_risk"].map(urgency_of)
    alert = alert.sort_values("stockout_risk", ascending=False)

    st.markdown(
        f'<div class="fs-section-eyebrow">Early Warning System</div>'
        f'<div class="fs-section-title">{len(alert)} SKUs flagged \u2014 '
        f'{inr_short(alert["sales_at_risk_inr"].sum())} sales at risk</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        tbl = pd.DataFrame({
            "SKU": alert["sku_id"],
            "Item": alert["subcategory"],
            "Category": alert["category"],
            "Risk Score": alert["stockout_risk"].map(lambda v: f"{v * 100:.0f}%"),
            "Days Cover": alert["days_cover"].round(0).astype(int).astype(str) + "d",
            "Weekly Vel.": alert["avg_weekly_demand"].round(0).astype(int).astype(str) + " / wk",
            "Sales at Risk": alert["sales_at_risk_inr"].map(inr_short),
            "Demand Trend": alert["trend_pct"].map(lambda v: pct_str(v)),
            "Alert": alert["alert_pill"],
        })
        st.dataframe(
            render_table(tbl, pill_cols={"Alert": ALERT_CONFIG},
                         mono_cols=["SKU", "Risk Score", "Days Cover", "Weekly Vel.", "Sales at Risk", "Demand Trend"]),
            use_container_width=True,
            hide_index=True,
            height=420,
        )
        download_csv(alert[["sku_id", "category", "subcategory", "stockout_risk", "stockout_cover_weeks",
                            "avg_weekly_demand", "sales_at_risk_inr", "trend_pct"]],
                     "stockout_alerts.csv", "\u2b07 Export Alerts")

    with right:
        st.markdown(
            '<div class="fs-section-eyebrow">Risk Distribution</div>'
            '<div class="fs-section-title">Flagged SKUs by Category</div>',
            unsafe_allow_html=True,
        )
        by_cat = alert.groupby("category").size().reset_index(name="count").sort_values("count")
        fig = go.Figure(go.Bar(
            y=by_cat["category"], x=by_cat["count"], orientation="h",
            marker_color=["#F59E0B", "#A78BFA", "#38BDF8", "#FB923C", "#34D399"][: len(by_cat)],
            hovertemplate="%{y}: %{x} SKUs<extra></extra>",
        ))
        fig.update_xaxes(title=dict(text="SKU count"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(base_layout(fig, height=200), use_container_width=True)

        st.markdown(
            '<div class="fs-section-eyebrow">Risk Thresholds</div>',
            unsafe_allow_html=True,
        )
        for label, lo, color in [("Critical", 0.9, "#F87171"), ("High", 0.75, "#FB923C"), ("Medium", 0.5, "#F59E0B")]:
            cnt = int((alert["stockout_risk"] >= lo).sum())
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0">'
                f'<span style="font-size:12px;color:{APP_THEME["text_body"]}">'
                f'<span style="color:{color};margin-right:6px">\u25cf</span>{label} '
                f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;color:{APP_THEME["text_muted"]}">{lo * 100:.0f}\u2013100%</span></span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:12px;color:{color};font-weight:600">{cnt} SKUs</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Markdown Candidates
# ─────────────────────────────────────────────────────────────────────────────

def suggested_discount(overstock_risk: float) -> float:
    if overstock_risk >= 0.9:
        return 0.25
    if overstock_risk >= 0.75:
        return 0.20
    if overstock_risk >= 0.6:
        return 0.15
    return 0.10


def md_action_of(discount: float) -> str:
    if discount >= 0.2:
        return "Markdown"
    if discount >= 0.15:
        return "Flash Sale"
    return "Monitor"


def page_markdown(filtered: pd.DataFrame) -> None:
    md = filtered[filtered["quadrant"] == "MARKDOWN / CLEAR"].copy()
    if md.empty:
        st.success("No overstocked SKUs flagged. \U0001f389")
        return

    md["discount"] = md["overstock_risk"].map(suggested_discount)
    md["new_price"] = md["list_price"] * (1 - md["discount"])
    md["action"] = md["discount"].map(md_action_of)

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card_html("Capital Locked \u2014 Overstock", inr_group(md["locked_capital_inr"].sum()),
                              None, None, f"{len(md)} SKUs flagged for action", "#A78BFA"), unsafe_allow_html=True)
    c2.markdown(kpi_card_html("Avg Overstock Cover", f"{md['overstock_cover_weeks'].mean():.0f} weeks",
                              None, None, "time in excess inventory", "#FB923C"), unsafe_allow_html=True)
    c3.markdown(kpi_card_html("Projected Recovery", inr_group(md["locked_capital_inr"].sum() * 0.8),
                              None, None, "if all markdowns applied", "#34D399"), unsafe_allow_html=True)

    st.markdown(
        '<div class="fs-section-eyebrow">Markdown Candidates</div>'
        '<div class="fs-section-title">Overstock SKUs \u2014 recommended clearance actions</div>',
        unsafe_allow_html=True,
    )

    tbl = pd.DataFrame({
        "SKU": md["sku_id"],
        "Item": md["subcategory"],
        "Category": md["category"],
        "Overstock Score": md["overstock_risk"].map(lambda v: f"{v * 100:.0f}%"),
        "Excess Units": md["excess_units"].astype(int).astype(str),
        "Current Price": md["list_price"].map(inr_group),
        "Suggested Discount": md["discount"].map(lambda v: f"-{v * 100:.0f}%"),
        "New Price": md["new_price"].map(inr_group),
        "Capital Locked": md["locked_capital_inr"].map(inr_group),
        "Action": md["action"],
    })
    st.dataframe(
        render_table(tbl, pill_cols={"Action": MD_ACTION_CONFIG},
                     mono_cols=["SKU", "Overstock Score", "Excess Units", "Current Price",
                                "Suggested Discount", "New Price", "Capital Locked"]),
        use_container_width=True,
        hide_index=True,
        height=360,
    )
    download_csv(md[["sku_id", "category", "subcategory", "overstock_risk", "excess_units",
                     "list_price", "discount", "new_price", "locked_capital_inr"]],
                 "markdown_candidates.csv", "\u2b07 Export List")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def build_sidebar(weekly: pd.DataFrame, risk: pd.DataFrame) -> dict:
    st.sidebar.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
        '<span style="width:8px;height:8px;background:#F59E0B;border-radius:50%"></span>'
        '<span class="fs-brand">NORTHBAY LIVING</span></div>'
        f'<div style="font-family:Instrument Sans,sans-serif;font-size:12px;color:{APP_THEME["text_muted"]};margin-bottom:14px">'
        'Project FORESIGHT \u2014 Demand &amp; Inventory Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.radio(
        "Appearance",
        options=("dark", "light"),
        index=0 if get_theme_base() == "dark" else 1,
        format_func=lambda base: ("\u2600\ufe0f  Light" if base == "light" else "\U0001F319  Dark"),
        key="app_theme",
        horizontal=True,
        help="Switch between the dark and light dashboard theme. Changes save to the URL.",
    )
    sync_theme_to_url()

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Scoring Service")

    api_url = st.sidebar.text_input(
        "API URL", value=st.session_state.get("api_url", "") or DEFAULT_API_URL, key="api_url"
    )
    if DEFAULT_AUTH_TOKEN:
        st.sidebar.text_input(
            "API Token", value=DEFAULT_AUTH_TOKEN, type="password", key="api_token"
        )
        st.sidebar.caption("Token injected from FORESIGHT_AUTH_TOKEN.")
    api_col, badge_col = st.sidebar.columns([1, 1])
    if api_col.button("Check service"):
        svc = check_api()

    status = st.session_state.get("api_status", "◇ Not checked")
    ok = st.session_state.get("api_ok")
    color = "#34D399" if ok else ("#F87171" if st.session_state.get("api_checked") else "#6B7494")
    badge_col.markdown(
        f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{color};'
        f'background:{APP_THEME["soft"]};border:1px solid {APP_THEME["border"]};'
        f'border-radius:3px;padding:4px 8px;white-space:nowrap">{status}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Fallback: reads give local parquet automatically when offline.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Filters")

    all_cats = sorted(weekly["category"].dropna().unique().tolist())
    selected_cats = st.sidebar.multiselect("Category", options=all_cats, default=all_cats)

    all_quads = ["REORDER NOW", "MARKDOWN / CLEAR", "WATCH / VOLATILE", "HEALTHY"]
    selected_quads = st.sidebar.multiselect("Risk Quadrant", options=all_quads, default=all_quads)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Model Info")
    fc = load_forecast()
    if "model_type" in fc.columns and not fc.empty:
        st.sidebar.markdown(f"- Model: `{fc['model_type'].iloc[0]}`")
        st.sidebar.markdown(f"- Horizon: `{int(fc.groupby('sku_id')['week_start'].count().max())} weeks`")
        st.sidebar.markdown(f"- Tracked SKUs: `{int(risk['sku_id'].nunique())}`")
    st.sidebar.caption("Project FORESIGHT v1.0 \u00b7 Zidio Development")

    return {"cats": selected_cats, "quads": selected_quads}


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global APP_THEME
    init_theme_from_url()
    APP_THEME = current_theme()
    inject_css()

    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0 4px">'
        '<div style="display:flex;align-items:center;gap:12px">'
        '<span style="width:8px;height:8px;background:#F59E0B;border-radius:50%"></span>'
        '<span class="fs-brand">NORTHBAY LIVING</span>'
        f'<span style="width:1px;height:16px;background:{APP_THEME["border_strong"]}"></span>'
        f'<span style="font-family:Instrument Sans,sans-serif;font-size:12px;color:{APP_THEME["text_muted"]}">'
        'Project FORESIGHT \u2014 Demand &amp; Inventory Intelligence</span></div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:{APP_THEME["text_muted"]}">'
        f'{datetime.now().strftime("%b %d, %Y")}</div></div>',
        unsafe_allow_html=True,
    )

    if not _check_data():
        st.error(
            "**No processed data found.**\n\nRun the pipeline first:\n"
            "```bash\n"
            "python data/generate_data.py\n"
            "python src/pipeline.py\n"
            "python src/forecast.py\n"
            "python src/risk.py\n"
            "```",
            icon="\u26a0\ufe0f",
        )
        return

    with st.spinner("Loading data\u2026"):
        weekly = load_weekly()
        forecast = load_forecast()
        risk = load_risk()

    filters = build_sidebar(weekly, risk)
    filtered = risk[
        risk["category"].isin(filters["cats"]) & risk["quadrant"].isin(filters["quads"])
    ].copy()

    tabs = st.tabs(["Planning Dashboard", "Reorder Plan", "Stockout Alerts", "Markdown Candidates"])

    with tabs[0]:
        page_dashboard(weekly, forecast, risk, filtered)
    with tabs[1]:
        page_reorder(filtered)
    with tabs[2]:
        page_stockout(weekly, filtered)
    with tabs[3]:
        page_markdown(filtered)


if __name__ == "__main__":
    main()