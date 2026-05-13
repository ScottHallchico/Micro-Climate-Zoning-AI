"""Phase 5: Governance API — FastAPI REST service for permit compliance checking.

Endpoints:
  POST /v1/permit/check — Check permit against active zoning directive
  GET  /v1/directive/{id}/provenance — Full provenance record
  GET  /v1/health — Health check
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.shared.types import ComplianceResponse, ZoningDirective
from src.shared.config import PINNConfig
from src.provenance.store import ProvenanceStore
from src.zoning.generator import ZoningGenerator
from src.cfd.runner import CFDRunner
from src.pinn.model import PINNModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Micro-Climate Zoning AI — Governance API",
    description="REST API for building permit compliance checking against physics-grounded zoning codes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (production would use PostGIS)
_directives_store: dict[str, dict[str, Any]] = {}
_api_keys: set[str] = {"dev-key-001", "test-key-001"}
_audit_log: list[dict[str, Any]] = []
_provenance_store = ProvenanceStore()
_cfd_runner = CFDRunner(provenance_store=_provenance_store)
_pinn_model: PINNModel | None = None
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ACTUAL_WEATHER_PATH = DATA_DIR / "actual_weather_measurements.csv"
ACTUAL_BLOCKS_PATH = DATA_DIR / "actual_blocks.csv"

PINN_FEATURE_NAMES = [
    "svf",
    "lambda_p",
    "hw_ratio",
    "albedo",
    "green_cover",
    "height_norm",
    "uhi",
    "wind",
    "solar",
    "surface_temp",
]


# ---------------------------------------------------------------------------
# Pydantic models for request/response validation
# ---------------------------------------------------------------------------

class PermitCheckRequest(BaseModel):
    """Building permit application payload."""
    block_id: str = Field(..., description="The block identifier", min_length=1)
    proposed_height_m: float = Field(..., description="Proposed building height in meters", gt=0)
    proposed_far: float = Field(..., description="Proposed Floor Area Ratio", gt=0)
    proposed_roof_albedo: float = Field(..., description="Proposed roof albedo", ge=0, le=1)
    proposed_green_cover_pct: float = Field(..., description="Proposed green cover percentage", ge=0, le=100)


class PermitCheckResponse(BaseModel):
    """Compliance check result."""
    compliance_status: str = Field(..., description="compliant | non_compliant | conditional")
    violated_fields: list[str] = Field(default_factory=list)
    active_directive_version_id: str = Field(...)
    checked_at: str = Field(...)


class ProvenanceResponse(BaseModel):
    """Full provenance record for a directive."""
    artifact_id: str
    artifact_type: str
    producing_component: str
    input_artifact_ids: list[str]
    creation_timestamp: str
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    directives_count: int
    timestamp: str


class CFDBlockInput(BaseModel):
    """Dashboard block payload for reduced-order CFD/solar simulation."""
    id: str
    zone_class: str
    max_height_m: float = Field(..., gt=0)
    uhi_intensity: float = Field(default=0.0)
    svf: float = Field(default=0.5, ge=0, le=1)
    lambda_p: float = Field(default=0.35, ge=0, le=1)
    hw_ratio: float = Field(default=1.4, ge=0)
    albedo: float = Field(default=0.28, ge=0, le=1)
    green_cover: float = Field(default=20.0, ge=0, le=100)


class CFDSimulationRequest(BaseModel):
    """Reduced-order CFD and solar simulation request."""
    blocks: list[CFDBlockInput] | None = None
    wind_speed_ms: float | None = Field(default=None, gt=0, le=30)
    wind_direction_deg: float | None = Field(default=None, ge=0, le=360)
    sun_altitude_deg: float = Field(default=58.0, ge=1, le=89)
    sun_azimuth_deg: float = Field(default=135.0, ge=0, le=360)
    weather_mode: str = Field(default="design_peak")


class CFDSimulationResponse(BaseModel):
    """Reduced-order CFD and solar simulation response."""
    simulation_type: str
    model: str
    generated_at: str
    block_count: int
    results: list[dict[str, Any]]


class PINNBlockInput(BaseModel):
    """Dashboard block payload for PINN inference."""
    id: str
    zone_class: str = Field(default="BASELINE_UNCHANGED")
    max_height_m: float = Field(..., gt=0)
    uhi_intensity: float = Field(default=0.0)
    svf: float = Field(default=0.5, ge=0, le=1)
    lambda_p: float = Field(default=0.35, ge=0, le=1)
    hw_ratio: float = Field(default=1.4, ge=0)
    albedo: float = Field(default=0.28, ge=0, le=1)
    green_cover: float = Field(default=20.0, ge=0, le=100)
    wind_speed: float | None = Field(default=None, ge=0)
    solar_radiation_wm2: float | None = Field(default=None, ge=0)
    surface_temp_c: float | None = None


class PINNPredictionRequest(BaseModel):
    """Run PINN surrogate prediction for dashboard blocks."""
    blocks: list[PINNBlockInput]


class PINNPredictionResponse(BaseModel):
    """PINN prediction response for the 3D dashboard."""
    simulation_type: str
    model: str
    generated_at: str
    block_count: int
    results: list[dict[str, Any]]


class ActualDataResponse(BaseModel):
    """Actual data payload used by dashboard and simulation endpoints."""
    data_source: str
    generated_at: str
    weather: dict[str, Any]
    block_count: int
    blocks: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Authentication middleware
# ---------------------------------------------------------------------------

def _authenticate(x_api_key: str | None) -> str:
    """Validate API key and return the key identifier."""
    if not x_api_key or x_api_key not in _api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide a valid X-API-Key header.",
        )
    # Return identifier (masked key), not the actual key value
    return f"key-{x_api_key[:4]}***"


def _log_audit(
    block_id: str,
    api_key_id: str,
    compliance_outcome: str,
) -> None:
    """Log a permit check to the audit trail."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "block_id": block_id,
        "api_key_identifier": api_key_id,
        "compliance_outcome": compliance_outcome,
    }
    _audit_log.append(entry)
    logger.info(f"Audit: {entry}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing actual data file: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _load_actual_weather(mode: str = "design_peak") -> dict[str, Any]:
    rows = _read_csv(ACTUAL_WEATHER_PATH)
    if not rows:
        raise HTTPException(status_code=404, detail="actual_weather_measurements.csv has no rows")

    def heat_score(row: dict[str, str]) -> float:
        return (
            _as_float(row.get("temperature_c")) * 1.5
            + _as_float(row.get("shortwave_radiation_wm2")) / 80.0
            + _as_float(row.get("soil_temperature_0cm_c")) * 0.8
        )

    if mode == "latest":
        selected = rows[-1]
    else:
        selected = max(rows, key=heat_score)

    return {
        "timestamp": selected["timestamp"],
        "source": selected["source"],
        "source_latitude": _as_float(selected["source_latitude"]),
        "source_longitude": _as_float(selected["source_longitude"]),
        "project_latitude": _as_float(selected["project_latitude"]),
        "project_longitude": _as_float(selected["project_longitude"]),
        "elevation_m": _as_float(selected["elevation_m"]),
        "temperature_c": _as_float(selected["temperature_c"]),
        "relative_humidity_pct": _as_float(selected["relative_humidity_pct"]),
        "wind_speed_ms": _as_float(selected["wind_speed_ms"]),
        "wind_direction_deg": _as_float(selected["wind_direction_deg"]),
        "shortwave_radiation_wm2": _as_float(selected["shortwave_radiation_wm2"]),
        "surface_pressure_hpa": _as_float(selected["surface_pressure_hpa"]),
        "soil_temperature_0cm_c": _as_float(selected["soil_temperature_0cm_c"]),
        "selection_mode": mode,
    }


def _classify_actual_zone(row: dict[str, str]) -> str:
    roads = int(_as_float(row.get("osm_road_count")))
    green = _as_float(row.get("green_cover_pct"))
    plan_area = _as_float(row.get("lambda_p"))
    height = _as_float(row.get("max_height_m"))
    if roads >= 15:
        return "WIND_CORRIDOR_CRITICAL"
    if green <= 7 and plan_area >= 0.38:
        return "THERMAL_REMEDIATION"
    if height >= 18 and green >= 10:
        return "DENSITY_ADAPTIVE"
    return "BASELINE_UNCHANGED"


def _build_actual_blocks(weather: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _read_csv(ACTUAL_BLOCKS_PATH)
    blocks: list[dict[str, Any]] = []
    for row in rows:
        green_cover = _as_float(row["green_cover_pct"])
        albedo = _as_float(row["albedo"])
        plan_area = _as_float(row["lambda_p"])
        height = _as_float(row["max_height_m"])
        hw_ratio = _as_float(row["hw_ratio"])
        morphology_heat = plan_area * 3.8 + hw_ratio * 0.42 + max(0.0, height - 15.0) * 0.035
        cooling = green_cover * 0.025 + albedo * 1.2
        surface_temp = weather["soil_temperature_0cm_c"] + morphology_heat - cooling
        air_temp = weather["temperature_c"] + morphology_heat * 0.28 - green_cover * 0.012
        uhi = max(0.0, surface_temp - weather["temperature_c"])
        blocks.append({
            "id": row["block_id"],
            "col": int(_as_float(row["col"])),
            "row": int(_as_float(row["row"])),
            "zone_class": _classify_actual_zone(row),
            "max_height_m": round(height, 1),
            "uhi_intensity": round(uhi, 2),
            "temperature": round(273.15 + surface_temp, 2),
            "svf": round(_as_float(row["svf"]), 2),
            "lambda_p": round(plan_area, 2),
            "hw_ratio": round(hw_ratio, 1),
            "albedo": round(albedo, 2),
            "green_cover": round(green_cover),
            "wind_speed": weather["wind_speed_ms"],
            "solar_radiation_wm2": weather["shortwave_radiation_wm2"],
            "surface_temp_c": round(surface_temp, 1),
            "air_temp_c": round(air_temp, 1),
            "osm_building_count": int(_as_float(row["osm_building_count"])),
            "osm_road_count": int(_as_float(row["osm_road_count"])),
            "osm_green_feature_count": int(_as_float(row["osm_green_feature_count"])),
            "measured_avg_building_levels": row["measured_avg_building_levels"] or None,
            "data_source": "open_meteo_weather_plus_osm_morphology",
            "interventions": [],
        })
    return blocks


def _get_demo_pinn_model() -> PINNModel:
    """Lazy-load a lightweight PINN instance for untrained dashboard inference."""
    global _pinn_model
    if _pinn_model is None:
        config = PINNConfig(encoder_layers=2, encoder_width=64, max_epochs=5)
        model = PINNModel(input_dim=10, config=config, provenance_store=_provenance_store)
        model._training_mean = torch.tensor(
            [0.50, 0.35, 0.40, 0.28, 0.30, 0.50, 0.35, 0.55, 0.55, 0.72],
            dtype=torch.float32,
        )
        model._training_std = torch.tensor(
            [0.22, 0.18, 0.24, 0.18, 0.20, 0.28, 0.30, 0.24, 0.25, 0.16],
            dtype=torch.float32,
        )
        model._feature_names = PINN_FEATURE_NAMES
        model._training_dataset_version = "untrained-pinn-actual-inputs-v1"
        model.eval()
        _pinn_model = model
    return _pinn_model


def _block_to_pinn_features(block: PINNBlockInput) -> torch.Tensor:
    """Convert dashboard block fields into the 10-feature PINN input vector."""
    wind_speed = block.wind_speed if block.wind_speed is not None else 3.0
    solar = block.solar_radiation_wm2 if block.solar_radiation_wm2 is not None else 520.0
    surface_temp = block.surface_temp_c if block.surface_temp_c is not None else 35.5 + block.uhi_intensity
    return torch.tensor(
        [
            block.svf,
            block.lambda_p,
            min(block.hw_ratio / 4.0, 1.5),
            block.albedo,
            block.green_cover / 100.0,
            min(block.max_height_m / 60.0, 1.5),
            block.uhi_intensity / 6.0,
            min(wind_speed / 8.0, 1.5),
            min(solar / 1000.0, 1.5),
            surface_temp / 50.0,
        ],
        dtype=torch.float32,
    )


def _interpret_pinn_prediction(
    block: PINNBlockInput,
    wind_field: np.ndarray,
    energy_field: np.ndarray,
    ood_warning: bool,
    ood_features: list[str],
) -> dict[str, Any]:
    """Turn raw PINN head outputs into stable dashboard-scale quantities.

    Without a trained checkpoint, the raw neural outputs are blended with
    physics-inspired terms so the UI shows a useful surrogate instead of
    arbitrary untrained numbers.
    """
    wind = np.tanh(wind_field.flatten())
    energy = np.tanh(energy_field.flatten())
    base_surface_temp = block.surface_temp_c if block.surface_temp_c is not None else 35.5 + block.uhi_intensity
    green_cooling = block.green_cover * 0.018
    albedo_cooling = max(0.0, block.albedo - 0.25) * 2.1
    height_penalty = max(0.0, block.hw_ratio - 1.0) * 0.35 + block.max_height_m * 0.015
    solar_penalty = ((block.solar_radiation_wm2 or 520.0) - 520.0) / 260.0
    neural_adjustment = float(energy[0] * 1.2)
    delta_temp = float(np.clip(block.uhi_intensity * 0.35 + height_penalty + solar_penalty - green_cooling - albedo_cooling + neural_adjustment, -2.5, 5.5))
    wind_factor = float(np.clip(0.54 + wind[0] * 0.16 + (block.green_cover / 100.0) * 0.08 - block.lambda_p * 0.12, 0.25, 0.95))
    height_sensitivity = float(np.clip(0.10 + block.hw_ratio * 0.08 + energy[1] * 0.22, -0.4, 0.9))
    green_sensitivity = float(np.clip(-0.08 - block.green_cover * 0.004 - abs(energy[2]) * 0.18, -0.65, -0.03))
    confidence = 0.56 if ood_warning else float(np.clip(0.78 - abs(energy[3]) * 0.08, 0.62, 0.86))
    return {
        "block_id": block.id,
        "pinn": {
            "predicted_delta_temp_c": round(delta_temp, 2),
            "predicted_surface_temp_c": round(base_surface_temp + delta_temp, 1),
            "predicted_wind_factor": round(wind_factor, 2),
            "height_sensitivity_c_per_10m": round(height_sensitivity, 2),
            "green_cover_sensitivity_c_per_10pct": round(green_sensitivity, 2),
            "confidence": round(confidence, 2),
            "ood_warning": ood_warning,
            "ood_features": ood_features,
            "training_dataset_version": "untrained-pinn-actual-inputs-v1",
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/permit/check", response_model=PermitCheckResponse)
async def check_permit(
    request: PermitCheckRequest,
    x_api_key: str | None = Header(None),
) -> PermitCheckResponse:
    """Check a building permit application against the active zoning directive.

    Returns compliance status, list of violated fields, and the active directive version.
    """
    api_key_id = _authenticate(x_api_key)

    # Look up active directive for the block
    directive_data = _directives_store.get(request.block_id)
    if directive_data is None:
        _log_audit(request.block_id, api_key_id, "not_found")
        raise HTTPException(
            status_code=404,
            detail=f"No active zoning directive found for block_id '{request.block_id}'",
        )

    # Evaluate compliance
    violated_fields: list[str] = []

    # Check height
    max_height = directive_data.get("max_height_m")
    if max_height is not None and request.proposed_height_m > max_height:
        violated_fields.append("proposed_height_m")

    # Check green cover
    min_green = 15.0  # 15% minimum from constraints
    for intervention in directive_data.get("mandatory_interventions", []):
        if intervention.get("type") in ("green_roof", "street_tree_canopy"):
            coverage = intervention.get("min_coverage_pct", 0)
            if coverage:
                min_green = max(min_green, coverage)
    if request.proposed_green_cover_pct < min_green:
        violated_fields.append("proposed_green_cover_pct")

    # Check albedo
    for intervention in directive_data.get("mandatory_interventions", []):
        if intervention.get("type") == "pavement_albedo":
            min_albedo = intervention.get("min_albedo", 0)
            if min_albedo and request.proposed_roof_albedo < min_albedo:
                violated_fields.append("proposed_roof_albedo")

    # Determine status
    if not violated_fields:
        status = "compliant"
    elif directive_data.get("zone_class") == "DENSITY_ADAPTIVE" and len(violated_fields) == 1:
        status = "conditional"
    else:
        status = "non_compliant"

    _log_audit(request.block_id, api_key_id, status)

    return PermitCheckResponse(
        compliance_status=status,
        violated_fields=violated_fields,
        active_directive_version_id=directive_data.get("version_id", "unknown"),
        checked_at=datetime.utcnow().isoformat(),
    )


@app.get("/v1/directive/{directive_version_id}/provenance", response_model=ProvenanceResponse)
async def get_provenance(
    directive_version_id: str,
    x_api_key: str | None = Header(None),
) -> ProvenanceResponse:
    """Return the full provenance record for a directive."""
    _authenticate(x_api_key)

    record = _provenance_store.get_record(directive_version_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No provenance record found for directive '{directive_version_id}'",
        )

    return ProvenanceResponse(
        artifact_id=record.artifact_id,
        artifact_type=record.artifact_type,
        producing_component=record.producing_component,
        input_artifact_ids=record.input_artifact_ids,
        creation_timestamp=record.creation_timestamp.isoformat(),
        metadata=record.metadata,
    )


@app.get("/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint — no authentication required."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        directives_count=len(_directives_store),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/v1/data/actual", response_model=ActualDataResponse)
async def get_actual_data(mode: str = "design_peak") -> ActualDataResponse:
    """Return actual public data converted into dashboard-ready block values."""
    weather = _load_actual_weather(mode=mode)
    blocks = _build_actual_blocks(weather)
    return ActualDataResponse(
        data_source="open_meteo_weather_plus_openstreetmap",
        generated_at=datetime.utcnow().isoformat(),
        weather=weather,
        block_count=len(blocks),
        blocks=blocks,
    )


@app.post("/v1/cfd/microclimate", response_model=CFDSimulationResponse)
async def simulate_microclimate(request: CFDSimulationRequest) -> CFDSimulationResponse:
    """Run fast block-level wind and sunlight simulation for the dashboard.

    This endpoint is a reduced-order CFD/solar model intended for interactive
    planning previews. Production deployment would replace this with OpenFOAM
    wind fields and a full radiative-transfer/shadow pipeline.
    """
    weather = _load_actual_weather(mode=request.weather_mode)
    block_payload = [block.model_dump() for block in request.blocks] if request.blocks else _build_actual_blocks(weather)
    results = _cfd_runner.simulate_block_microclimate(
        blocks=block_payload,
        wind_speed_ms=request.wind_speed_ms or weather["wind_speed_ms"],
        wind_direction_deg=request.wind_direction_deg or weather["wind_direction_deg"],
        sun_altitude_deg=request.sun_altitude_deg,
        sun_azimuth_deg=request.sun_azimuth_deg,
        measured_solar_radiation_wm2=weather["shortwave_radiation_wm2"],
        base_air_temp_c=weather["temperature_c"],
        surface_temp_reference_c=weather["soil_temperature_0cm_c"],
    )
    return CFDSimulationResponse(
        simulation_type="wind_and_solar",
        model="actual_weather_osm_reduced_order_cfd_solar_v1",
        generated_at=datetime.utcnow().isoformat(),
        block_count=len(results),
        results=results,
    )


@app.post("/v1/pinn/predict", response_model=PINNPredictionResponse)
async def predict_with_pinn(request: PINNPredictionRequest) -> PINNPredictionResponse:
    """Run PINN inference for dashboard blocks.

    This uses the project's PINN architecture without a real trained checkpoint.
    It is intended to prove the API/UI integration path now; trained CFD or
    field-measurement data can replace the demo weights later.
    """
    model = _get_demo_pinn_model()
    results: list[dict[str, Any]] = []
    for block in request.blocks:
        features = _block_to_pinn_features(block)
        prediction = model.predict(features)
        results.append(
            _interpret_pinn_prediction(
                block=block,
                wind_field=prediction.wind_field,
                energy_field=prediction.energy_field,
                ood_warning=prediction.ood_warning,
                ood_features=prediction.ood_features,
            )
        )
    return PINNPredictionResponse(
        simulation_type="pinn_surrogate",
        model="untrained_pinn_actual_inputs_v1",
        generated_at=datetime.utcnow().isoformat(),
        block_count=len(results),
        results=results,
    )


# ---------------------------------------------------------------------------
# Directive management (internal API for loading directives)
# ---------------------------------------------------------------------------

def load_directives(directives: list[ZoningDirective]) -> int:
    """Load zoning directives into the in-memory store.

    In production, this would query PostGIS.
    Returns the number of directives loaded.
    """
    for d in directives:
        _directives_store[d.block_id] = ZoningGenerator.serialize_directive(d)
        _directives_store[d.block_id]["version_id"] = d.version_id
    return len(directives)


def get_audit_log() -> list[dict[str, Any]]:
    """Return the audit log entries."""
    return list(_audit_log)


def clear_stores() -> None:
    """Clear all in-memory stores. For testing only."""
    _directives_store.clear()
    _audit_log.clear()
