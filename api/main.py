"""
Predix — FastAPI Scoring API
────────────────────────────────────────────────────────────────────────────────
Serves real-time churn and upsell scores from pre-computed Parquet files.

Endpoints:
  GET  /health                   — Liveness probe
  GET  /api/v1/score/churn/{uid} — Churn probability + risk tier for a user
  GET  /api/v1/score/upsell/{uid}— Top-K upsell category recommendations
  GET  /api/v1/segment/{uid}     — RFM segment + profile for a user
  POST /admin/reload-scores      — Reload Parquet files into memory (Airflow hook)
  GET  /api/v1/metrics/overview  — Aggregate dashboard KPIs (JSON)

Run locally:
    uvicorn api.main:app --reload --port 8000
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger("predix_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT         = Path(__file__).parent.parent
SCORES_DIR   = ROOT / "pipeline" / "scoring" / "output"
SCORING_DIR  = ROOT / "pipeline" / "scoring"
API_TOKEN    = os.getenv("PREDIX_API_TOKEN", "predix-dev-token")

# ── In-memory score store ─────────────────────────────────────────────────────
_store: dict[str, pd.DataFrame] = {}


def _load_scores() -> None:
    for name, path in [
        ("churn",  SCORES_DIR / "churn_scores.parquet"),
        ("upsell", SCORES_DIR / "upsell_scores.parquet"),
        ("rfm",    SCORING_DIR / "rfm.parquet"),
    ]:
        if path.exists():
            _store[name] = pd.read_parquet(path)
            log.info("Loaded %s: %d rows.", name, len(_store[name]))
        else:
            log.warning("Score file not found: %s (run train_and_register.py)", path)
            _store[name] = pd.DataFrame()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _load_scores()
    yield


app = FastAPI(
    title       = "Predix Intelligence API",
    description = "Churn prediction & upsell scoring for e-commerce",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChurnScore(BaseModel):
    user_id:     int
    churn_score: float   = Field(..., ge=0, le=1, description="Churn probability [0–1]")
    churn_risk:  str     = Field(..., description="Low | Medium | High")

class UpsellRecommendation(BaseModel):
    category:     str
    upsell_score: float
    opportunity:  float
    affinity:     float

class UpsellResponse(BaseModel):
    user_id:          int
    recommendations:  list[UpsellRecommendation]

class UserSegment(BaseModel):
    user_id:    int
    segment:    str
    rfm_score:  int
    r_score:    int
    f_score:    int
    m_score:    int
    recency:    int
    frequency:  int
    monetary:   float


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok", "scores_loaded": {k: len(v) for k, v in _store.items()}}


# ── Churn score ───────────────────────────────────────────────────────────────
@app.get("/api/v1/score/churn/{user_id}", response_model=ChurnScore, tags=["Scoring"])
def get_churn_score(user_id: int) -> ChurnScore:
    df = _store.get("churn", pd.DataFrame())
    if df.empty:
        raise HTTPException(
            status_code=503, detail="Churn scores not loaded. Run batch scoring first."
        )

    row = df[df["user_id"] == user_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in churn scores.")

    r = row.iloc[0]
    return ChurnScore(
        user_id     = user_id,
        churn_score = float(r["churn_score"]),
        churn_risk  = str(r.get("churn_risk", "Unknown")),
    )


# ── Upsell recommendations ────────────────────────────────────────────────────
@app.get("/api/v1/score/upsell/{user_id}", response_model=UpsellResponse, tags=["Scoring"])
def get_upsell_score(user_id: int, top_k: int = 5) -> UpsellResponse:
    df = _store.get("upsell", pd.DataFrame())
    if df.empty:
        raise HTTPException(status_code=503, detail="Upsell scores not loaded.")

    rows = df[df["user_id"] == user_id].sort_values("upsell_score", ascending=False).head(top_k)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in upsell scores.")

    return UpsellResponse(
        user_id = user_id,
        recommendations = [
            UpsellRecommendation(
                category     = str(r["category"]),
                upsell_score = float(r["upsell_score"]),
                opportunity  = float(r.get("opportunity", 0)),
                affinity     = float(r.get("affinity", 0)),
            )
            for _, r in rows.iterrows()
        ],
    )


# ── RFM segment ───────────────────────────────────────────────────────────────
@app.get("/api/v1/segment/{user_id}", response_model=UserSegment, tags=["Segments"])
def get_user_segment(user_id: int) -> UserSegment:
    df = _store.get("rfm", pd.DataFrame())
    if df.empty:
        raise HTTPException(status_code=503, detail="RFM data not loaded.")

    row = df[df["user_id"] == user_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    r = row.iloc[0]
    return UserSegment(
        user_id   = user_id,
        segment   = str(r.get("segment", "Unknown")),
        rfm_score = int(r.get("rfm_score", 0)),
        r_score   = int(r.get("r_score", 0)),
        f_score   = int(r.get("f_score", 0)),
        m_score   = int(r.get("m_score", 0)),
        recency   = int(r.get("recency", 0)),
        frequency = int(r.get("frequency", 0)),
        monetary  = float(r.get("monetary", 0)),
    )


# ── Bulk scores ───────────────────────────────────────────────────────────────
@app.get("/api/v1/scores/top-churn", tags=["Scoring"])
def top_churn_users(limit: int = 100) -> list[dict[str, Any]]:
    df = _store.get("churn", pd.DataFrame())
    if df.empty:
        raise HTTPException(status_code=503, detail="Churn scores not loaded.")
    top = df.sort_values("churn_score", ascending=False).head(limit)
    return top[["user_id", "churn_score", "churn_risk"]].to_dict(orient="records")


# ── Admin: reload scores ──────────────────────────────────────────────────────
@app.post("/admin/reload-scores", tags=["Admin"])
def reload_scores(authorization: str = Header(default="")) -> dict:
    token = authorization.replace("Bearer ", "").strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token.")
    _load_scores()
    return {"status": "reloaded", "rows": {k: len(v) for k, v in _store.items()}}


# ── Overview KPIs ─────────────────────────────────────────────────────────────
@app.get("/api/v1/metrics/overview", tags=["Metrics"])
def overview_kpis() -> dict[str, Any]:
    rfm   = _store.get("rfm",   pd.DataFrame())
    churn = _store.get("churn", pd.DataFrame())

    return {
        "total_users":       int(len(rfm)),
        "churned_users":     int((churn["churn_risk"] == "High").sum()) if not churn.empty else 0,
        "avg_churn_score":   float(churn["churn_score"].mean()) if not churn.empty else 0,
        "segment_breakdown": rfm["segment"].value_counts().to_dict() if not rfm.empty else {},
    }
