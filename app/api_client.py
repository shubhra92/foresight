"""
api_client.py
=============
Project FORESIGHT — thin client for the FastAPI scoring service
(service/main.py). Used by the Streamlit dashboard to fetch live
/summary, /skus and /predict responses.

Design notes
------------
* Every call is wrapped so the dashboard degrades gracefully when the
  service is offline — the caller falls back to the local parquet
  artefacts instead of crashing.
* `check_health()` should be called before live reads so the status
  badge in the sidebar reflects reality.

Usage
-----
    client = ScoringService(base_url="http://localhost:8000")
    if client.check_health():
        summary = client.summary()
        result  = client.predict(["SKU0001", "SKU0042"], horizon_weeks=8)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

DEFAULT_API_URL = os.environ.get("FORESIGHT_API_URL", "http://localhost:8000")
DEFAULT_AUTH_TOKEN = os.environ.get("FORESIGHT_AUTH_TOKEN", "") or None
DEFAULT_TIMEOUT = 5.0


class ScoringServiceError(RuntimeError):
    """Raised when the scoring service is unreachable or returns an error."""


class ScoringService:
    """Minimal client for the FORESIGHT FastAPI scoring service."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        token: Optional[str] = DEFAULT_AUTH_TOKEN,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.token = token
        self.healthy = False
        self.data_loaded = False
        self.last_error: Optional[str] = None

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, **kwargs: Any) -> Any:
        try:
            resp = requests.get(
                f"{self.base_url}{path}", timeout=self.timeout,
                headers=self._headers(), **kwargs,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 — any failure = service offline
            raise ScoringServiceError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, json: dict[str, Any]) -> Any:
        try:
            resp = requests.post(
                f"{self.base_url}{path}", json=json, timeout=self.timeout,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ScoringServiceError(f"POST {path} failed: {exc}") from exc

    # ── health / status ──────────────────────────────────────────────────────

    def check_health(self) -> bool:
        """Ping /health and update internal status. Returns True if ready."""
        try:
            health = self._get("/health")
            self.healthy = bool(health.get("status") == "ok")
            self.data_loaded = bool(health.get("data_loaded", False))
            self.last_error = health.get("error")
        except ScoringServiceError as exc:
            self.healthy = False
            self.data_loaded = False
            self.last_error = str(exc)
        return self.healthy and self.data_loaded

    def status_text(self) -> str:
        if self.healthy and self.data_loaded:
            return "● Online"
        if not self.base_url:
            return "○ Not configured"
        return "○ Offline"

    # ── endpoints ────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Portfolio-level KPIs from GET /summary."""
        if not self.check_health():
            raise ScoringServiceError(self.last_error or "Service offline.")
        return self._get("/summary")

    def evaluation(self) -> dict[str, Any]:
        """Rolling-origin backtest evidence (WAPE vs baseline) from GET /evaluation."""
        if not self.check_health():
            raise ScoringServiceError(self.last_error or "Service offline.")
        return self._get("/evaluation")

    def list_skus(self) -> list[dict[str, Any]]:
        """All available SKUs from GET /skus."""
        if not self.check_health():
            raise ScoringServiceError(self.last_error or "Service offline.")
        payload = self._get("/skus")
        return payload.get("skus", [])

    def predict(
        self,
        sku_ids: list[str],
        horizon_weeks: int = 8,
    ) -> dict[str, Any]:
        """Live forecast + risk from POST /predict."""
        if not self.check_health():
            raise ScoringServiceError(self.last_error or "Service offline.")
        return self._post(
            "/predict",
            {"sku_ids": list(sku_ids), "horizon_weeks": int(horizon_weeks)},
        )