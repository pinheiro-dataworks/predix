"""Contract tests for the FastAPI scoring service.

Score files are written to a temporary directory so the tests exercise the
real startup path (Parquet → in-memory store) without the training pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main

CHURN = pd.DataFrame({
    "user_id": [101, 102, 103],
    "churn_score": [0.12, 0.48, 0.91],
    "churn_risk": ["Low", "Medium", "High"],
})
UPSELL = pd.DataFrame({
    "user_id": [101, 101, 101, 102],
    "category": ["electronics", "appliances", "apparel", "computers"],
    "upsell_score": [0.40, 0.75, 0.10, 0.55],
    "opportunity": [0.5, 0.9, 0.1, 0.6],
    "affinity": [0.2, 0.4, 0.0, 0.3],
})
RFM = pd.DataFrame({
    "user_id": [101, 102],
    "segment": ["Champions", "At Risk"],
    "rfm_score": [14, 7],
    "r_score": [5, 2],
    "f_score": [5, 3],
    "m_score": [4, 2],
    "recency": [2, 25],
    "frequency": [9, 3],
    "monetary": [1234.5, 210.0],
})


@pytest.fixture
def score_dirs(tmp_path, monkeypatch):
    scoring_dir = tmp_path / "scoring"
    output_dir = scoring_dir / "output"
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(api_main, "SCORING_DIR", scoring_dir)
    monkeypatch.setattr(api_main, "SCORES_DIR", output_dir)
    monkeypatch.setattr(api_main, "_store", {})
    return scoring_dir, output_dir


@pytest.fixture
def client(score_dirs):
    scoring_dir, output_dir = score_dirs
    CHURN.to_parquet(output_dir / "churn_scores.parquet")
    UPSELL.to_parquet(output_dir / "upsell_scores.parquet")
    RFM.to_parquet(scoring_dir / "rfm.parquet")
    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture
def empty_client(score_dirs):
    with TestClient(api_main.app) as c:
        yield c


def test_health_reports_loaded_row_counts(client):
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json() == {"status": "ok", "scores_loaded": {"churn": 3, "upsell": 4, "rfm": 2}}


def test_churn_score_contract(client):
    r = client.get("/api/v1/score/churn/103")

    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"user_id", "churn_score", "churn_risk"}
    assert body == {"user_id": 103, "churn_score": 0.91, "churn_risk": "High"}


def test_unknown_user_returns_404(client):
    for path in ["/api/v1/score/churn/999", "/api/v1/score/upsell/999", "/api/v1/segment/999"]:
        assert client.get(path).status_code == 404, path


def test_non_integer_user_id_is_rejected(client):
    assert client.get("/api/v1/score/churn/abc").status_code == 422


def test_upsell_returns_top_k_sorted_by_score(client):
    r = client.get("/api/v1/score/upsell/101", params={"top_k": 2})

    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert [rec["category"] for rec in recs] == ["appliances", "electronics"]
    assert set(recs[0]) == {"category", "upsell_score", "opportunity", "affinity"}


def test_segment_contract(client):
    r = client.get("/api/v1/segment/101")

    assert r.status_code == 200
    body = r.json()
    assert body["segment"] == "Champions"
    assert set(body) == {
        "user_id", "segment", "rfm_score", "r_score", "f_score", "m_score",
        "recency", "frequency", "monetary",
    }


def test_top_churn_is_ordered_and_limited(client):
    r = client.get("/api/v1/scores/top-churn", params={"limit": 2})

    assert r.status_code == 200
    assert [row["user_id"] for row in r.json()] == [103, 102]


def test_overview_aggregates(client):
    body = client.get("/api/v1/metrics/overview").json()

    assert body["total_users"] == 2
    assert body["churned_users"] == 1  # users in the High risk tier
    assert body["avg_churn_score"] == pytest.approx(CHURN["churn_score"].mean())
    assert body["segment_breakdown"] == {"Champions": 1, "At Risk": 1}


def test_missing_score_files_return_503(empty_client):
    for path in ["/api/v1/score/churn/101", "/api/v1/score/upsell/101", "/api/v1/segment/101"]:
        assert empty_client.get(path).status_code == 503, path


def test_overview_works_without_scores(empty_client):
    r = empty_client.get("/api/v1/metrics/overview")

    assert r.status_code == 200
    assert r.json()["total_users"] == 0


def test_reload_requires_token(client):
    assert client.post("/admin/reload-scores").status_code == 401
    assert client.post(
        "/admin/reload-scores", headers={"Authorization": "Bearer wrong"}
    ).status_code == 401


def test_reload_picks_up_new_files(client, score_dirs, monkeypatch):
    monkeypatch.setattr(api_main, "API_TOKEN", "test-token")
    _, output_dir = score_dirs
    pd.concat([CHURN, CHURN.assign(user_id=104)]).to_parquet(output_dir / "churn_scores.parquet")

    r = client.post("/admin/reload-scores", headers={"Authorization": "Bearer test-token"})

    assert r.status_code == 200
    assert r.json()["rows"]["churn"] == 6
    assert client.get("/api/v1/score/churn/104").status_code == 200
