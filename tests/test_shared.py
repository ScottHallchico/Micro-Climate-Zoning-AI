"""Tests for shared types and provenance store."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.shared.types import (
    UDTVoxel, UDTArtifact, UCMBlock, CFDSample,
    CheckpointManifest, ZoningDirective, MandatoryIntervention,
    DirectiveProvenance, PermitApplication, ComplianceResponse,
    ProvenanceRecord, WardHeatIndex, WardMap,
)
from src.shared.exceptions import (
    IngestionValidationError, CheckpointError,
    InfeasibilityError, UCMValidationError,
)
from src.provenance.store import ProvenanceStore


class TestSharedTypes:
    """Test all shared data model constructors and field defaults."""

    def test_udt_voxel_creation(self):
        voxel = UDTVoxel(
            x=10.0, y=20.0, z=5.0,
            geometry_type="building",
            albedo=0.3, emissivity=0.9,
            heat_capacity=1.5e6,
            boundary_temp_k=300.0,
            material_class="concrete",
        )
        assert voxel.geometry_type == "building"
        assert 0 <= voxel.albedo <= 1
        assert 0 <= voxel.emissivity <= 1
        assert voxel.heat_capacity > 0

    def test_udt_artifact_auto_version_id(self):
        artifact = UDTArtifact()
        assert artifact.version_id is not None
        assert len(artifact.version_id) > 0
        assert artifact.voxel_resolution_m == 2.0

    def test_ucm_block_canyon_classes(self):
        for canyon in ["deep_canyon", "regular_canyon", "shallow_canyon", "transitional_zone"]:
            block = UCMBlock(
                block_id="BLK-01", udt_version_id="v1",
                svf=0.5, lambda_p=0.3, lambda_f=0.2,
                hw_ratio=1.5, z0=0.5, canyon_class=canyon,
            )
            assert block.canyon_class == canyon

    def test_zoning_directive_creation(self):
        directive = ZoningDirective(
            block_id="BLK-04",
            zone_class="WIND_CORRIDOR_CRITICAL",
            max_height_m=12.0,
            height_justification="Test justification",
            permitted_uses=["residential_low"],
            mandatory_interventions=[],
            compliance_deadline_months=12,
            review_trigger="Annual review",
        )
        assert directive.zone_class == "WIND_CORRIDOR_CRITICAL"
        assert directive.max_height_m == 12.0

    def test_ward_heat_index_flagging(self):
        whi = WardHeatIndex(
            ward_id="W-01",
            total_cooling_benefit_c_m2=5000.0,
            total_retrofit_cost=100000.0,
            displacement_risk_score=0.7,
            equity_review_flagged=True,
        )
        assert whi.equity_review_flagged is True
        assert whi.displacement_risk_score > 0.5


class TestExceptions:
    """Test custom exception hierarchy."""

    def test_ingestion_validation_error(self):
        err = IngestionValidationError(
            "satellite", "Missing emissivity metadata",
            {"path": "/data/thermal.tif"},
        )
        assert "satellite" in str(err)
        assert err.stream_name == "satellite"
        assert err.affected_extent is not None

    def test_checkpoint_error(self):
        err = CheckpointError("/models/bad.pt", "Corrupt file")
        assert "bad.pt" in str(err)
        assert err.path == "/models/bad.pt"

    def test_infeasibility_error(self):
        err = InfeasibilityError(
            ["FAR + green_cover > area", "Budget exceeded"],
            {"block": "BLK-05"},
        )
        assert len(err.conflicting_constraints) == 2
        assert err.details is not None

    def test_ucm_validation_error(self):
        err = UCMValidationError("BLK-01", "building_height", -5.0, "[0, 600] m")
        assert err.block_id == "BLK-01"
        assert err.value == -5.0


class TestProvenanceStore:
    """Test provenance store CRUD and lineage tracing."""

    def _make_store(self, tmp_path: Path) -> ProvenanceStore:
        return ProvenanceStore(store_path=str(tmp_path / "prov.json"))

    def test_record_and_retrieve(self, tmp_path):
        store = self._make_store(tmp_path)
        record = ProvenanceRecord(
            artifact_id="art-001",
            artifact_type="udt",
            producing_component="ingestion_pipeline",
            input_artifact_ids=[],
            metadata={"voxels": 1000},
        )
        store.record_artifact(record)

        retrieved = store.get_record("art-001")
        assert retrieved is not None
        assert retrieved.artifact_type == "udt"
        assert retrieved.metadata["voxels"] == 1000

    def test_lineage_tracing(self, tmp_path):
        store = self._make_store(tmp_path)

        # Create a chain: udt -> ucm -> cfd -> pinn
        store.record_artifact(ProvenanceRecord(
            artifact_id="udt-1", artifact_type="udt",
            producing_component="ingestion", input_artifact_ids=[],
        ))
        store.record_artifact(ProvenanceRecord(
            artifact_id="ucm-1", artifact_type="ucm",
            producing_component="ucm_builder", input_artifact_ids=["udt-1"],
        ))
        store.record_artifact(ProvenanceRecord(
            artifact_id="cfd-1", artifact_type="cfd_dataset",
            producing_component="cfd_runner", input_artifact_ids=["ucm-1"],
        ))

        lineage = store.get_lineage("cfd-1")
        assert len(lineage) == 3
        assert lineage[0].artifact_id == "udt-1"
        assert lineage[-1].artifact_id == "cfd-1"

    def test_get_nonexistent_record(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.get_record("nonexistent") is None

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "prov.json")
        store1 = ProvenanceStore(store_path=path)
        store1.record_artifact(ProvenanceRecord(
            artifact_id="test-1", artifact_type="udt",
            producing_component="test", input_artifact_ids=[],
        ))

        store2 = ProvenanceStore(store_path=path)
        assert store2.get_record("test-1") is not None

    def test_get_artifacts_by_type(self, tmp_path):
        store = self._make_store(tmp_path)
        store.record_artifact(ProvenanceRecord(
            artifact_id="a1", artifact_type="udt",
            producing_component="test", input_artifact_ids=[],
        ))
        store.record_artifact(ProvenanceRecord(
            artifact_id="a2", artifact_type="pinn_checkpoint",
            producing_component="test", input_artifact_ids=[],
        ))

        udts = store.get_artifacts_by_type("udt")
        assert len(udts) == 1
        assert udts[0].artifact_id == "a1"
