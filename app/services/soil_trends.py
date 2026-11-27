"""Historical Soil Trend Analysis Service (Not an ML Forecast).

Computes chronological earliest-to-latest parameter changes across existing
soil_tests records for an authenticated farmer's field(s), enforcing strict
unit compatibility and safe handling of missing or non-numeric values.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

from ..models import SoilTest
from ..schemas import ParameterTrendOut, SoilTrendResponseOut, TrendPointOut

_PARAMETER_SPECS: list[tuple[str, str, str, bool]] = [
    ("nitrogen", "Nitrogen (N)", "kg/ha", True),
    ("phosphorus", "Phosphorus (P)", "kg/ha", True),
    ("potassium", "Potassium (K)", "kg/ha", True),
    ("ph", "Soil pH", "pH", False),
    ("organic_matter", "Organic Matter (OM)", "%", False),
    ("moisture", "Soil Moisture", "%", False),
    ("health_score", "Soil Health Score", "score", False),
]


def _finite_float(val: object) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _compute_safe_health_score(t: SoilTest) -> float | None:
    ph = _finite_float(getattr(t, "ph", None))
    n = _finite_float(getattr(t, "nitrogen", None))
    p = _finite_float(getattr(t, "phosphorus", None))
    k = _finite_float(getattr(t, "potassium", None))
    om = _finite_float(getattr(t, "organic_matter", None))
    m = _finite_float(getattr(t, "moisture", None))
    if None in (ph, n, p, k, om, m):
        return None
    ph_score = 1.0 - min(abs(ph - 6.5), 3.5) / 3.5
    n_score = min(n / 280.0, 1.0)
    p_score = min(p / 55.0, 1.0)
    k_score = min(k / 280.0, 1.0)
    om_score = min(om / 3.0, 1.0)
    m_score = max(0.0, 1.0 - abs(m - 50.0) / 50.0)
    score = (
        ph_score * 25
        + n_score * 20
        + p_score * 15
        + k_score * 15
        + om_score * 15
        + m_score * 10
    )
    return round(score, 2)


def _normalize_unit(raw_unit: object) -> str | None:
    if not isinstance(raw_unit, str):
        return None
    cleaned = raw_unit.strip()
    return cleaned if cleaned else None


def compute_historical_soil_trends(
    tests: Iterable[SoilTest],
    field_id: int | None = None,
    field_name: str | None = None,
    health_score_fn: Callable[[SoilTest], float] | None = None,
) -> SoilTrendResponseOut:
    """Compute chronological earliest-to-latest trends from comparable soil-test records."""
    ordered = sorted(
        list(tests),
        key=lambda t: (t.tested_at, t.id),
    )
    total_records = len(ordered)

    if total_records == 0:
        return SoilTrendResponseOut(
            field_id=field_id,
            field_name=field_name,
            total_records=0,
            earliest_tested_at=None,
            latest_tested_at=None,
            status="insufficient_data",
            message="No historical soil test records found for trend analysis.",
            parameters=[
                ParameterTrendOut(
                    parameter=param,
                    label=label,
                    unit=expected_unit,
                    status="insufficient_data",
                    comparable_points=0,
                    note="No historical records available.",
                )
                for param, label, expected_unit, _ in _PARAMETER_SPECS
            ],
        )

    param_results: list[ParameterTrendOut] = []
    any_comparable = False

    for param, label, expected_unit, check_unit in _PARAMETER_SPECS:
        valid_points: list[TrendPointOut] = []
        incompatible_unit_found: str | None = None

        for t in ordered:
            if param == "health_score":
                val = _compute_safe_health_score(t)
                if val is None and health_score_fn is not None:
                    try:
                        val = _finite_float(health_score_fn(t))
                    except Exception:
                        val = None
            else:
                val = _finite_float(getattr(t, param, None))

            if val is None:
                continue

            if check_unit:
                raw_unit = getattr(t, f"{param}_unit", expected_unit)
                if raw_unit is None:
                    raw_unit = expected_unit
                norm_unit = _normalize_unit(raw_unit)
                if norm_unit is None or norm_unit.lower() != expected_unit.lower():
                    incompatible_unit_found = norm_unit or "unspecified"
                    break
                point_unit = expected_unit
            else:
                point_unit = expected_unit

            valid_points.append(
                TrendPointOut(
                    test_id=t.id,
                    tested_at=t.tested_at,
                    value=round(val, 2),
                    unit=point_unit,
                )
            )

        if incompatible_unit_found is not None:
            param_results.append(
                ParameterTrendOut(
                    parameter=param,
                    label=label,
                    unit=None,
                    status="incompatible_units",
                    comparable_points=0,
                    note=(
                        f"Incompatible or unknown unit ('{incompatible_unit_found}') detected; "
                        f"expected '{expected_unit}'. Trend withheld to avoid misleading comparisons."
                    ),
                    points=[],
                )
            )
            continue

        if len(valid_points) < 2:
            single_val = valid_points[0].value if len(valid_points) == 1 else None
            param_results.append(
                ParameterTrendOut(
                    parameter=param,
                    label=label,
                    unit=expected_unit,
                    status="insufficient_data",
                    earliest_value=single_val,
                    latest_value=single_val,
                    comparable_points=len(valid_points),
                    note="At least 2 valid comparable records are required to compute a historical trend.",
                    points=valid_points,
                )
            )
            continue

        any_comparable = True
        earliest_val = valid_points[0].value
        latest_val = valid_points[-1].value
        change = round(latest_val - earliest_val, 2)
        pct_change = (
            round(((latest_val - earliest_val) / abs(earliest_val)) * 100.0, 2)
            if abs(earliest_val) > 1e-9
            else None
        )
        if abs(change) <= 1e-6:
            direction = "stable"
        elif change > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        param_results.append(
            ParameterTrendOut(
                parameter=param,
                label=label,
                unit=expected_unit,
                status=direction,
                earliest_value=earliest_val,
                latest_value=latest_val,
                change=change,
                percent_change=pct_change,
                comparable_points=len(valid_points),
                points=valid_points,
            )
        )

    overall_status = "available" if any_comparable else "insufficient_data"
    overall_msg = (
        f"Historical changes computed across {total_records} chronological records."
        if any_comparable
        else "At least 2 comparable historical records are required to display parameter trends."
    )

    return SoilTrendResponseOut(
        field_id=field_id,
        field_name=field_name,
        total_records=total_records,
        earliest_tested_at=ordered[0].tested_at,
        latest_tested_at=ordered[-1].tested_at,
        status=overall_status,
        message=overall_msg,
        parameters=param_results,
    )

