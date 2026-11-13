"""Soil test CRUD + advisory engine.

Health score formula (0–100):
  Scores each parameter on how close it is to the ideal range, then takes a
  weighted average:

    ph_score        = 1.0 - min(|ph - 6.5|, 3.5) / 3.5          (ideal ~6.5)
    nitrogen_score  = min(nitrogen / 280, 1.0)                    (ideal ≥280 kg/ha)
    phosphorus_score= min(phosphorus / 55, 1.0)                   (ideal ≥55 kg/ha)
    potassium_score = min(potassium / 280, 1.0)                   (ideal ≥280 kg/ha)
    om_score        = min(organic_matter / 3.0, 1.0)              (ideal ≥3 %)
    moisture_score  = 1.0 - |moisture - 50| / 50                  (ideal ~50 %)

  health_score = round(
      (ph_score * 25 + nitrogen_score * 20 + phosphorus_score * 15
       + potassium_score * 15 + om_score * 15 + moisture_score * 10)
      , 2)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func as sa_func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Field, SoilTest
from ..schemas import AdvisoryOut, SoilTestIn, SoilTestOut

router = APIRouter(tags=["soil-tests"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_health_score(t: SoilTest) -> float:
    """Return a 0–100 soil health score (see module docstring for formula)."""
    ph_score = 1.0 - min(abs(t.ph - 6.5), 3.5) / 3.5
    n_score  = min(t.nitrogen   / 280.0, 1.0)
    p_score  = min(t.phosphorus / 55.0,  1.0)
    k_score  = min(t.potassium  / 280.0, 1.0)
    om_score = min(t.organic_matter / 3.0, 1.0)
    m_score  = max(0.0, 1.0 - abs(t.moisture - 50.0) / 50.0)

    score = (
        ph_score  * 25
        + n_score * 20
        + p_score * 15
        + k_score * 15
        + om_score * 15
        + m_score * 10
    )
    return round(score, 2)


def _make_advisory(test: SoilTest, score: float) -> AdvisoryOut:
    """Rule-based crop advisory from soil parameters."""
    recs: list[str] = []
    crops: list[str] = []
    warnings: list[str] = []

    # --- pH guidance ---
    if test.ph < 5.5:
        recs.append("Apply agricultural lime to raise pH; soil is too acidic.")
        warnings.append(f"pH {test.ph:.1f} is severely acidic — most crops will struggle.")
    elif test.ph < 6.0:
        recs.append("Consider light liming to bring pH into the 6.0–7.0 range.")
    elif test.ph > 8.0:
        recs.append("Apply sulfur or gypsum to lower pH; soil is alkaline.")
        warnings.append(f"pH {test.ph:.1f} is alkaline — micronutrient lock-up is likely.")

    # --- Nitrogen ---
    if test.nitrogen < 100:
        recs.append("Apply nitrogen fertiliser (urea or ammonium nitrate) before planting.")
        warnings.append("Nitrogen is very low — expect stunted growth.")
    elif test.nitrogen < 200:
        recs.append("Supplement with moderate nitrogen; include a legume cover crop to fix N naturally.")

    # --- Phosphorus ---
    if test.phosphorus < 20:
        recs.append("Apply superphosphate or DAP to boost phosphorus for root development.")
        warnings.append("Critical phosphorus deficiency — seedling establishment will be poor.")
    elif test.phosphorus < 40:
        recs.append("Moderate phosphorus application recommended.")

    # --- Potassium ---
    if test.potassium < 100:
        recs.append("Apply muriate of potash (MOP) to build potassium reserves.")
        warnings.append("Low potassium increases susceptibility to disease and drought stress.")
    elif test.potassium < 200:
        recs.append("Light potassium top-dressing is beneficial.")

    # --- Organic matter ---
    if test.organic_matter < 1.0:
        recs.append("Incorporate compost or farmyard manure to build organic matter.")
        warnings.append("Very low organic matter — soil structure and water retention are poor.")
    elif test.organic_matter < 2.0:
        recs.append("Add compost/manure annually to improve soil health long-term.")

    # --- Moisture ---
    if test.moisture < 20:
        recs.append("Irrigate before planting; soil is very dry.")
        warnings.append("Low soil moisture — seed germination will be unreliable.")
    elif test.moisture > 80:
        recs.append("Improve drainage (raised beds or tile drains) to avoid waterlogging.")
        warnings.append("Excess moisture — risk of root rot and anaerobic conditions.")

    # --- Crop suitability (simple rule set) ---
    ph = test.ph
    n  = test.nitrogen
    p  = test.phosphorus

    if 5.5 <= ph <= 6.5 and n >= 150 and p >= 30:
        crops.extend(["Rice", "Maize", "Sugarcane"])
    if 6.0 <= ph <= 7.5 and n >= 100:
        crops.extend(["Wheat", "Barley", "Oats"])
    if 5.8 <= ph <= 6.8 and test.organic_matter >= 1.5:
        crops.extend(["Potato", "Tomato", "Pepper"])
    if 6.0 <= ph <= 7.0:
        crops.extend(["Soybean", "Groundnut", "Chickpea"])
    if ph < 6.0:
        crops.extend(["Blueberry", "Tea", "Pineapple"])
    if ph > 7.0:
        crops.extend(["Barley", "Alfalfa", "Asparagus"])
    if not crops:
        crops.append("Consult a local agronomist — conditions are outside standard thresholds.")

    # deduplicate preserving order
    seen: set[str] = set()
    unique_crops = [c for c in crops if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    return AdvisoryOut(
        soil_test_id=test.id,
        health_score=score,
        recommendations=recs,
        suitable_crops=unique_crops,
        warnings=warnings,
    )


def _enrich(obj: SoilTest) -> SoilTestOut:
    out = SoilTestOut.model_validate(obj)
    out.health_score = _compute_health_score(obj)
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/fields/{field_id}/soil-tests", response_model=SoilTestOut, status_code=201)
def create_soil_test(field_id: int, body: SoilTestIn, db: Session = Depends(get_db)):
    """Stores laboratory soil parameters (N, P, K, pH, organic matter, moisture) for a field."""
    if not db.get(Field, field_id):
        raise HTTPException(404, "Field not found")
    obj = SoilTest(field_id=field_id, **body.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _enrich(obj)


@router.get("/fields/{field_id}/soil-tests", response_model=list[SoilTestOut])
def list_soil_tests(field_id: int, db: Session = Depends(get_db)):
    """History for one field, newest first (used later for degradation trend analysis)."""
    if not db.get(Field, field_id):
        raise HTTPException(404, "Field not found")
    rows = db.scalars(
        select(SoilTest)
        .where(SoilTest.field_id == field_id)
        .order_by(SoilTest.tested_at.desc(), SoilTest.id.desc())
    ).all()
    return [_enrich(r) for r in rows]


@router.get("/soil-tests", response_model=list[SoilTestOut])
def list_soil_tests_by_field(field_id: int, db: Session = Depends(get_db)):
    """GET /api/soil-tests/?field_id=X — flat query without field path parameter."""
    rows = db.scalars(
        select(SoilTest)
        .where(SoilTest.field_id == field_id)
        .order_by(SoilTest.tested_at.desc(), SoilTest.id.desc())
    ).all()
    return [_enrich(r) for r in rows]


@router.get("/soil-tests/{test_id}", response_model=SoilTestOut)
def get_soil_test(test_id: int, db: Session = Depends(get_db)):
    obj = db.get(SoilTest, test_id)
    if not obj:
        raise HTTPException(404, "Soil test not found")
    return _enrich(obj)


@router.get("/soil-tests/{test_id}/advisory", response_model=AdvisoryOut)
def get_advisory(test_id: int, db: Session = Depends(get_db)):
    """Rule-based crop recommendations derived from recorded soil parameters."""
    obj = db.get(SoilTest, test_id)
    if not obj:
        raise HTTPException(404, "Soil test not found")
    score = _compute_health_score(obj)
    return _make_advisory(obj, score)
