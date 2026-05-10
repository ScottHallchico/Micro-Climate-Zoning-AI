"""Tests for the UCM Builder (Phase 2)."""

from pathlib import Path

import pytest

from src.ucm.builder import UCMBuilder
from src.shared.types import UDTArtifact
from src.shared.exceptions import UCMValidationError
from src.provenance.store import ProvenanceStore


class TestUCMBuilder:
    """Test morphological index computation and canyon classification."""

    def _make_builder(self, tmp_path: Path) -> UCMBuilder:
        return UCMBuilder(
            provenance_store=ProvenanceStore(str(tmp_path / "prov.json"))
        )

    def test_canyon_boundary_hw_0_9_shallow(self, tmp_path):
        builder = self._make_builder(tmp_path)
        result = builder._classify_canyon(0.9, {})
        assert result == "shallow_canyon"

    def test_canyon_boundary_hw_1_0_regular(self, tmp_path):
        builder = self._make_builder(tmp_path)
        result = builder._classify_canyon(1.0, {})
        assert result == "regular_canyon"

    def test_canyon_boundary_hw_2_5_regular(self, tmp_path):
        builder = self._make_builder(tmp_path)
        result = builder._classify_canyon(2.5, {})
        assert result == "regular_canyon"

    def test_canyon_boundary_hw_2_6_deep(self, tmp_path):
        builder = self._make_builder(tmp_path)
        result = builder._classify_canyon(2.6, {})
        assert result == "deep_canyon"

    def test_transitional_zone_has_turbulence_params(self, tmp_path):
        builder = self._make_builder(tmp_path)
        udt = UDTArtifact()
        block_defs = [{
            "block_id": "BLK-T1",
            "building_heights": [10, 30, 5, 40],
            "block_area_m2": 5000,
            "footprint_area_m2": 2000,
            "mean_building_width_m": 15,
            "street_width_m": 12,
            "is_transitional": True,
        }]

        artifact = builder.build(udt, block_defs)
        # The block should be classified as transitional
        assert artifact.canyon_class_distribution.get("transitional_zone", 0) >= 1

    def test_invalid_height_negative_raises(self, tmp_path):
        builder = self._make_builder(tmp_path)
        with pytest.raises(UCMValidationError):
            builder._validate_block_inputs({
                "block_id": "BLK-BAD",
                "building_heights": [-5.0],
                "block_area_m2": 1000,
            })

    def test_invalid_height_over_600_raises(self, tmp_path):
        builder = self._make_builder(tmp_path)
        with pytest.raises(UCMValidationError):
            builder._validate_block_inputs({
                "block_id": "BLK-BAD",
                "building_heights": [601.0],
                "block_area_m2": 1000,
            })

    def test_invalid_block_area_zero_raises(self, tmp_path):
        builder = self._make_builder(tmp_path)
        with pytest.raises(UCMValidationError):
            builder._validate_block_inputs({
                "block_id": "BLK-BAD",
                "building_heights": [10.0],
                "block_area_m2": 0,
            })

    def test_invalid_block_area_negative_raises(self, tmp_path):
        builder = self._make_builder(tmp_path)
        with pytest.raises(UCMValidationError):
            builder._validate_block_inputs({
                "block_id": "BLK-BAD",
                "building_heights": [10.0],
                "block_area_m2": -100,
            })

    def test_svf_within_bounds(self, tmp_path):
        builder = self._make_builder(tmp_path)
        svf = builder._compute_svf(20.0, 10.0)
        assert 0 <= svf <= 1

    def test_build_produces_artifact_with_provenance(self, tmp_path):
        builder = self._make_builder(tmp_path)
        udt = UDTArtifact()
        artifact = builder.build(udt)
        assert artifact.version_id is not None
        assert artifact.block_count > 0
        assert sum(artifact.canyon_class_distribution.values()) == artifact.block_count
