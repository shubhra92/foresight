"""Functional tests for the FastAPI scoring service (service/main.py)."""

import pytest
from fastapi.testclient import TestClient
from service.main import app, AUTH_TOKEN, _store


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def test_store_loaded(client):
    """The in-memory store is populated at startup — confirm it's ready."""
    assert _store.ready, "Service data store failed to load; run the pipeline first."


def _headers(token=None):
    return {"Authorization": f"Bearer {token}"} if token else {}


# ─────────────────────────────────────────────────────────────────────────────
# Health (open — no auth required)
# ─────────────────────────────────────────────────────────────────────────────

def test_health_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["data_loaded"] is True
    assert isinstance(body["forecast_rows"], int)


def test_health_ignores_bad_token(client):
    r = client.get("/health", headers=_headers("wrong-token"))
    assert r.status_code == 200  # health is always open


# ─────────────────────────────────────────────────────────────────────────────
# Auth — when token is configured
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not AUTH_TOKEN, reason="AUTH_TOKEN not set in test env")
def test_protected_endpoint_rejects_no_token(client):
    r = client.get("/skus")
    assert r.status_code == 401
    assert "bearer" in r.json()["detail"].lower()


@pytest.mark.skipif(not AUTH_TOKEN, reason="AUTH_TOKEN not set in test env")
def test_protected_endpoint_rejects_bad_token(client):
    r = client.get("/skus", headers=_headers("nope"))
    assert r.status_code == 401


@pytest.mark.skipif(not AUTH_TOKEN, reason="AUTH_TOKEN not set in test env")
def test_protected_endpoint_accepts_valid_token(client):
    r = client.get("/skus", headers=_headers(AUTH_TOKEN))
    assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# /skus, /summary, /evaluation — shape checks
# ─────────────────────────────────────────────────────────────────────────────

def test_skus_shape(client):
    r = client.get("/skus")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    assert isinstance(body["skus"], list)
    assert "sku_id" in body["skus"][0]


def test_summary_shape(client):
    r = client.get("/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_skus"] > 0
    quadrants = body["reorder_count"] + body["markdown_count"] + body["watch_count"] + body["healthy_count"]
    assert quadrants == body["total_skus"]


def test_evaluation_shape(client):
    r = client.get("/evaluation")
    assert r.status_code == 200
    body = r.json()
    assert "wape_comparison" in body
    assert "lgbm" in body
    assert "baseline" in body
    assert body["wape_comparison"]["winner"] == "lgbm"
    assert "history" in body          # accuracy_tracking.parquet
    assert "last_refresh" in body     # reports/last_refresh.json


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_known_sku(client):
    r = client.post("/predict", json={"sku_ids": ["SKU0001"], "horizon_weeks": 8})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["sku_id"] == "SKU0001"
    assert len(body["results"][0]["forecast"]) == 8
    assert body["not_found"] == []


def test_predict_unknown_sku(client):
    r = client.post("/predict", json={"sku_ids": ["FAKE999"]})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
    assert body["not_found"] == ["FAKE999"]


def test_predict_horizon_out_of_range_returns_422(client):
    r = client.post("/predict", json={"sku_ids": ["SKU0001"], "horizon_weeks": 30})
    assert r.status_code == 422


def test_predict_empty_sku_ids_returns_422(client):
    r = client.post("/predict", json={"sku_ids": [], "horizon_weeks": 8})
    assert r.status_code == 422