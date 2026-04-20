"""
SIGHTAI – FastAPI Backend
Run:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os as _os; sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import json
import uuid
import sqlite3
import csv
import io
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

import config
from api.inference import predict_image, load_model, load_threshold

FRONTEND_DIR  = Path(__file__).parent.parent / "frontend"
DB_PATH       = config.SAVED_MODELS_DIR / "reviews.db"
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff"}


# ── Database setup ────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    config.SAVED_MODELS_DIR.mkdir(exist_ok=True)
    with _get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                review_id        TEXT PRIMARY KEY,
                timestamp        TEXT NOT NULL,
                filename         TEXT,
                ai_label         TEXT,
                ai_probability   REAL,
                ai_risk_level    TEXT,
                ai_threshold     REAL,
                patient_id       TEXT,
                patient_age      TEXT,
                patient_sex      TEXT,
                clinician_name   TEXT,
                clinician_role   TEXT,
                facility         TEXT,
                decision         TEXT,
                final_label      TEXT,
                clinical_notes   TEXT,
                recommended_action TEXT,
                urgency          TEXT
            )
        """)
        conn.commit()


def _migrate_json():
    """One-time migration from old reviews.json to SQLite."""
    old_path = config.SAVED_MODELS_DIR / "reviews.json"
    if not old_path.exists():
        return
    try:
        with open(old_path) as f:
            records = json.load(f)
        if not records:
            return
        fields = [
            "review_id","timestamp","filename","ai_label","ai_probability",
            "ai_risk_level","ai_threshold","patient_id","patient_age","patient_sex",
            "clinician_name","clinician_role","facility","decision","final_label",
            "clinical_notes","recommended_action","urgency",
        ]
        with _get_db() as conn:
            for r in records:
                conn.execute(
                    f"INSERT OR IGNORE INTO reviews ({','.join(fields)}) VALUES ({','.join(['?']*len(fields))})",
                    [r.get(f, "") for f in fields]
                )
            conn.commit()
        old_path.rename(old_path.with_suffix(".json.bak"))
        print(f"Migrated {len(records)} reviews from JSON to SQLite.")
    except Exception as e:
        print(f"JSON migration skipped: {e}")


def _save_review(record: dict):
    fields = list(record.keys())
    with _get_db() as conn:
        conn.execute(
            f"INSERT INTO reviews ({','.join(fields)}) VALUES ({','.join(['?']*len(fields))})",
            list(record.values())
        )
        conn.commit()


def _load_reviews(limit: int = 100, offset: int = 0,
                  risk: str = None, decision: str = None,
                  search: str = None) -> tuple[list, int]:
    where, params = [], []
    if risk:
        where.append("ai_risk_level = ?"); params.append(risk)
    if decision:
        where.append("decision = ?"); params.append(decision)
    if search:
        where.append("(patient_id LIKE ? OR clinician_name LIKE ? OR filename LIKE ? OR facility LIKE ?)")
        params.extend([f"%{search}%"] * 4)

    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with _get_db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM reviews {clause}", params).fetchone()[0]
        rows  = conn.execute(
            f"SELECT * FROM reviews {clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()
    return [dict(r) for r in rows], total


def _get_review_by_id(review_id: str) -> Optional[dict]:
    with _get_db() as conn:
        row = conn.execute(
            "SELECT * FROM reviews WHERE review_id = ?", [review_id.upper()]
        ).fetchone()
    return dict(row) if row else None


def _get_stats() -> dict:
    with _get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        if total == 0:
            return {
                "total_reviews": 0, "agreement_rate": 0,
                "override_rate": 0, "uncertain_rate": 0,
                "by_risk": {}, "by_decision": {}, "by_facility": [],
                "avg_probability": 0, "recent_7d": 0,
            }

        by_decision = {
            r["decision"]: r["cnt"]
            for r in conn.execute(
                "SELECT decision, COUNT(*) as cnt FROM reviews GROUP BY decision"
            ).fetchall()
        }
        by_risk = {
            r["ai_risk_level"]: r["cnt"]
            for r in conn.execute(
                "SELECT ai_risk_level, COUNT(*) as cnt FROM reviews GROUP BY ai_risk_level"
            ).fetchall()
        }
        top_facilities = [
            dict(r) for r in conn.execute(
                "SELECT facility, COUNT(*) as cnt FROM reviews "
                "WHERE facility != '' GROUP BY facility ORDER BY cnt DESC LIMIT 5"
            ).fetchall()
        ]
        avg_prob = conn.execute("SELECT AVG(ai_probability) FROM reviews").fetchone()[0] or 0
        recent_7d = conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE timestamp >= datetime('now','-7 days')"
        ).fetchone()[0]

    agree    = by_decision.get("agree", 0)
    disagree = by_decision.get("disagree", 0)
    uncertain= by_decision.get("uncertain", 0)

    return {
        "total_reviews":   total,
        "agreement_rate":  round(agree    / total * 100, 1),
        "override_rate":   round(disagree / total * 100, 1),
        "uncertain_rate":  round(uncertain/ total * 100, 1),
        "by_risk":         by_risk,
        "by_decision":     by_decision,
        "by_facility":     top_facilities,
        "avg_probability": round(avg_prob * 100, 1),
        "recent_7d":       recent_7d,
    }


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ClinicalReview(BaseModel):
    filename:        str
    ai_label:        str
    ai_probability:  float
    ai_risk_level:   str
    ai_threshold:    float
    patient_id:      Optional[str] = ""
    patient_age:     Optional[str] = ""
    patient_sex:     Optional[str] = ""
    clinician_name:  str
    clinician_role:  Optional[str] = ""
    facility:        Optional[str] = ""
    decision:        str
    final_label:     Optional[str] = ""
    clinical_notes:  Optional[str] = ""
    recommended_action: Optional[str] = ""
    urgency:         Optional[str] = ""


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _init_db()
    _migrate_json()
    if config.MODEL_PATH.exists():
        try:
            load_model()
            print(f"SIGHTAI model loaded | threshold={load_threshold():.4f}")
        except Exception as e:
            print(f"Warning: model pre-load failed — {e}")
    else:
        print(f"Warning: no model at {config.MODEL_PATH}. "
              "Run `python model/train.py` first.")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    lifespan=lifespan,
    title="SIGHTAI",
    description="AI-powered TB Detection – Ghana | Human-in-the-loop clinical review",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>SIGHTAI running. <a href='/docs'>API docs</a></h1>")


@app.get("/health")
async def health():
    ready = config.MODEL_PATH.exists()
    return {
        "status":      "ok",
        "model_ready": ready,
        "threshold":   load_threshold() if ready else None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(default=True,
                          description="Include Grad-CAM heatmap in response."),
):
    """Upload a chest X-ray → receive AI TB risk assessment + Grad-CAM explanation."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported type '{file.content_type}'.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded file is empty.")
    try:
        result = predict_image(image_bytes, explain=explain)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")
    return JSONResponse({"filename": file.filename, **result})


@app.post("/review")
async def submit_review(review: ClinicalReview):
    """Submit a clinician review (human-in-the-loop). Returns unique review_id."""
    record = {
        "review_id":  str(uuid.uuid4())[:8].upper(),
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        **review.model_dump(),
    }
    _save_review(record)
    return JSONResponse({
        "status":    "saved",
        "review_id": record["review_id"],
        "timestamp": record["timestamp"],
        "agreement": review.decision == "agree",
    })


@app.get("/reviews")
async def list_reviews(
    limit:    int = Query(default=20, le=100),
    offset:   int = Query(default=0),
    risk:     Optional[str] = Query(default=None, description="Filter by risk level: Low | Medium | High"),
    decision: Optional[str] = Query(default=None, description="Filter by decision: agree | disagree | uncertain"),
    search:   Optional[str] = Query(default=None, description="Search patient ID, clinician, file, facility"),
):
    """Return paginated clinician reviews with optional filters."""
    reviews, total = _load_reviews(limit=limit, offset=offset,
                                   risk=risk, decision=decision, search=search)
    return JSONResponse({"total": total, "limit": limit, "offset": offset, "reviews": reviews})


@app.get("/reviews/export/csv")
async def export_reviews_csv():
    """Download all reviews as a CSV file."""
    reviews, _ = _load_reviews(limit=10000)
    if not reviews:
        raise HTTPException(404, "No reviews to export.")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(reviews[0].keys()))
    writer.writeheader()
    writer.writerows(reviews)
    buf.seek(0)
    filename = f"sightai_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/reviews/{review_id}")
async def get_review(review_id: str):
    """Retrieve a specific review by ID."""
    record = _get_review_by_id(review_id)
    if not record:
        raise HTTPException(404, f"Review '{review_id}' not found.")
    return JSONResponse(record)


@app.get("/stats")
async def get_stats():
    """Aggregate statistics: agreement rate, risk distribution, override patterns."""
    return JSONResponse(_get_stats())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.main:app", host=config.API_HOST,
                port=config.API_PORT, reload=True)
