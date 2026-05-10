"""Tests for Governance API (Phase 5)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.governance.api import app, load_directives, clear_stores, get_audit_log
from src.shared.types import ZoningDirective, MandatoryIntervention, DirectiveProvenance


@pytest.fixture(autouse=True)
def _clean_stores():
    """Clear in-memory stores before each test."""
    clear_stores()
    yield
    clear_stores()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def _loaded_directive():
    """Load a sample directive for testing."""
    directive = ZoningDirective(
        block_id="BLK-04",
        zone_class="WIND_CORRIDOR_CRITICAL",
        max_height_m=12.0,
        height_justification="Corridor preservation test",
        permitted_uses=["residential_low"],
        mandatory_interventions=[
            MandatoryIntervention(
                type="pavement_albedo",
                cooling_effect_estimate_c=-0.8,
                min_albedo=0.5,
            ),
        ],
        compliance_deadline_months=12,
        review_trigger="Annual review",
        provenance=DirectiveProvenance(
            udt_version_id="udt-v1",
            pinn_checkpoint_version="pinn-v1",
            cfd_sample_ids=["cfd-001"],
            mobo_run_id="mobo-001",
        ),
        version_id="dir-v1",
    )
    load_directives([directive])
    return directive


class TestGovernanceAPI:
    """Test REST API endpoints for permit compliance."""

    def test_health_endpoint(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_unauthenticated_returns_401(self, client, _loaded_directive):
        resp = client.post("/v1/permit/check", json={
            "block_id": "BLK-04",
            "proposed_height_m": 10,
            "proposed_far": 2.0,
            "proposed_roof_albedo": 0.5,
            "proposed_green_cover_pct": 20,
        })
        assert resp.status_code == 401

    def test_unknown_block_returns_404(self, client, _loaded_directive):
        resp = client.post(
            "/v1/permit/check",
            json={
                "block_id": "NONEXISTENT",
                "proposed_height_m": 10,
                "proposed_far": 2.0,
                "proposed_roof_albedo": 0.5,
                "proposed_green_cover_pct": 20,
            },
            headers={"X-API-Key": "dev-key-001"},
        )
        assert resp.status_code == 404
        assert "NONEXISTENT" in resp.json()["detail"]

    def test_malformed_payload_returns_422(self, client):
        resp = client.post(
            "/v1/permit/check",
            json={"block_id": "BLK-04"},  # Missing required fields
            headers={"X-API-Key": "dev-key-001"},
        )
        assert resp.status_code == 422

    def test_compliant_application(self, client, _loaded_directive):
        resp = client.post(
            "/v1/permit/check",
            json={
                "block_id": "BLK-04",
                "proposed_height_m": 10,
                "proposed_far": 2.0,
                "proposed_roof_albedo": 0.6,
                "proposed_green_cover_pct": 30,
            },
            headers={"X-API-Key": "dev-key-001"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["compliance_status"] == "compliant"
        assert len(data["violated_fields"]) == 0

    def test_non_compliant_height(self, client, _loaded_directive):
        resp = client.post(
            "/v1/permit/check",
            json={
                "block_id": "BLK-04",
                "proposed_height_m": 20,  # Exceeds 12m cap
                "proposed_far": 2.0,
                "proposed_roof_albedo": 0.6,
                "proposed_green_cover_pct": 30,
            },
            headers={"X-API-Key": "dev-key-001"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["compliance_status"] == "non_compliant"
        assert "proposed_height_m" in data["violated_fields"]

    def test_audit_log_entry_written(self, client, _loaded_directive):
        client.post(
            "/v1/permit/check",
            json={
                "block_id": "BLK-04",
                "proposed_height_m": 10,
                "proposed_far": 2.0,
                "proposed_roof_albedo": 0.6,
                "proposed_green_cover_pct": 30,
            },
            headers={"X-API-Key": "dev-key-001"},
        )
        log = get_audit_log()
        assert len(log) >= 1
        assert log[-1]["block_id"] == "BLK-04"
        assert "key-" in log[-1]["api_key_identifier"]
        assert log[-1]["compliance_outcome"] == "compliant"
