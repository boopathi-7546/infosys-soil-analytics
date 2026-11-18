"""Soil test CRUD + advisory engine with farmer-specific authentication and data isolation.

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
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func as sa_func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Farmer, Field, SoilTest
from ..schemas import AdvisoryOut, SoilTestIn, SoilTestOut
from ..services.auth import get_current_farmer
from .fields import get_farmer_field_or_404

router = APIRouter(tags=["soil-tests"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_farmer_test_number_map(farmer_id: int, db: Session) -> dict[int, int]:
    """Return a mapping of {soil_test_id: sequential_test_number} for all tests belonging to this farmer.
    Numbering starts at 1 in chronological order (oldest is #1, newest is #N).
    """
    stmt = (
        select(SoilTest.id)
        .join(Field, SoilTest.field_id == Field.id)
        .where(Field.farmer_id == farmer_id)
        .order_by(SoilTest.id.asc())
    )
    test_ids = db.scalars(stmt).all()
    return {test_id: idx + 1 for idx, test_id in enumerate(test_ids)}


def _enrich(obj: SoilTest, test_number: int | None = None, db: Session | None = None) -> SoilTestOut:
    out = SoilTestOut.model_validate(obj)
    out.health_score = _compute_health_score(obj)
    if test_number is not None:
        out.test_number = test_number
        out.display_id = test_number
    elif db is not None and obj.field and obj.field.farmer_id is not None:
        seq = db.scalar(
            select(sa_func.count(SoilTest.id))
            .join(Field, SoilTest.field_id == Field.id)
            .where(Field.farmer_id == obj.field.farmer_id, SoilTest.id <= obj.id)
        ) or 1
        out.test_number = seq
        out.display_id = seq
    else:
        out.test_number = obj.id
        out.display_id = obj.id
    return out

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

    seen: set[str] = set()
    unique_crops = [c for c in crops if not (c in seen or seen.add(c))]

    return AdvisoryOut(
        soil_test_id=test.id,
        health_score=score,
        recommendations=recs,
        suitable_crops=unique_crops,
        warnings=warnings,
    )


def _get_farmer_soil_test_or_404(test_id: int, current_farmer: Farmer, db: Session) -> SoilTest:
    """Retrieve soil test verifying that the associated field belongs to the current farmer."""
    obj = db.get(SoilTest, test_id)
    if not obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soil test not found")
    # Verify ownership of the parent field
    if obj.field and obj.field.farmer_id is not None and obj.field.farmer_id != current_farmer.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soil test not found")
    return obj


# ---------------------------------------------------------------------------
# Routes (Authenticated)
# ---------------------------------------------------------------------------

@router.post("/fields/{field_id}/soil-tests", response_model=SoilTestOut, status_code=status.HTTP_201_CREATED)
def create_soil_test(
    field_id: int,
    body: SoilTestIn,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Stores laboratory soil parameters for a field owned by the authenticated farmer."""
    field = get_farmer_field_or_404(field_id, current_farmer, db)
    obj = SoilTest(field_id=field.id, **body.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _enrich(obj, db=db)


@router.get("/fields/{field_id}/soil-tests", response_model=list[SoilTestOut])
def list_soil_tests_for_field(
    field_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """History for one field (strictly owned by authenticated farmer), newest first."""
    field = get_farmer_field_or_404(field_id, current_farmer, db)
    num_map = _get_farmer_test_number_map(current_farmer.id, db)
    rows = db.scalars(
        select(SoilTest)
        .where(SoilTest.field_id == field.id)
        .order_by(SoilTest.tested_at.desc(), SoilTest.id.desc())
    ).all()
    return [_enrich(r, test_number=num_map.get(r.id), db=db) for r in rows]


@router.get("/soil-tests", response_model=list[SoilTestOut])
def list_soil_tests(
    field_id: int | None = Query(default=None, description="Optional field ID filter. If omitted, returns all soil tests for the authenticated farmer across all fields."),
    limit: int | None = Query(default=None, ge=1, description="Optional maximum number of records to return"),
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """GET /api/soil-tests — flat query returning all soil tests for the authenticated farmer, newest first."""
    num_map = _get_farmer_test_number_map(current_farmer.id, db)
    stmt = (
        select(SoilTest)
        .join(Field, SoilTest.field_id == Field.id)
        .where(Field.farmer_id == current_farmer.id)
    )
    if field_id is not None:
        get_farmer_field_or_404(field_id, current_farmer, db)
        stmt = stmt.where(SoilTest.field_id == field_id)

    stmt = stmt.order_by(SoilTest.tested_at.desc(), SoilTest.id.desc())
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = db.scalars(stmt).all()
    return [_enrich(r, test_number=num_map.get(r.id), db=db) for r in rows]


@router.get("/soil-tests/{test_id}", response_model=SoilTestOut)
def get_soil_test(
    test_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Retrieve single soil test (strictly owned by authenticated farmer)."""
    obj = _get_farmer_soil_test_or_404(test_id, current_farmer, db)
    num_map = _get_farmer_test_number_map(current_farmer.id, db)
    return _enrich(obj, test_number=num_map.get(obj.id), db=db)


@router.get("/soil-tests/{test_id}/advisory", response_model=AdvisoryOut)
def get_advisory(
    test_id: int,
    current_farmer: Farmer = Depends(get_current_farmer),
    db: Session = Depends(get_db),
):
    """Rule-based crop recommendations derived from recorded soil parameters (strictly owned)."""
    obj = _get_farmer_soil_test_or_404(test_id, current_farmer, db)
    score = _compute_health_score(obj)
    return _make_advisory(obj, score)
