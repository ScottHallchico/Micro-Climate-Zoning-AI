"""Phase 5: Governance API — FastAPI REST service for permit compliance checking.

Endpoints:
  POST /v1/permit/check — Check permit against active zoning directive
  GET  /v1/directive/{id}/provenance — Full provenance record
  GET  /v1/health — Health check
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.shared.types import ComplianceResponse, ZoningDirective
from src.provenance.store import ProvenanceStore
from src.zoning.generator import ZoningGenerator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Micro-Climate Zoning AI — Governance API",
    description="REST API for building permit compliance checking against physics-grounded zoning codes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (production would use PostGIS)
_directives_store: dict[str, dict[str, Any]] = {}
_api_keys: set[str] = {"dev-key-001", "test-key-001"}
_audit_log: list[dict[str, Any]] = []
_provenance_store = ProvenanceStore()


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
