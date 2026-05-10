"""Tests for the Ingestion Pipeline (Phase 1)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.shared.config import IngestionConfig
from src.shared.exceptions import IngestionValidationError
from src.provenance.store import ProvenanceStore


class TestIngestionPipeline:
    """Test LIDAR, satellite, meteorological, and material ingestion."""

    def _make_pipeline(self, tmp_path: Path) -> IngestionPipeline:
        return IngestionPipeline(
            config=IngestionConfig(),
            provenance_store=ProvenanceStore(str(tmp_path / "prov.json")),
        )

    def test_ingest_with_no_data_produces_artifact(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        artifact = pipeline.ingest()
        assert artifact.version_id is not None
        assert artifact.voxel_resolution_m == 2.0

    def test_csf_classification_output(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        rng = np.random.default_rng(42)
        points = np.column_stack([
            rng.uniform(0, 100, 1000),
            rng.uniform(0, 100, 1000),
            rng.uniform(0, 30, 1000),
        ])
        ground_mask = pipeline._csf_classify(points)
        assert ground_mask.dtype == bool
        assert ground_mask.sum() > 0  # Some ground points
        assert (~ground_mask).sum() > 0  # Some non-ground points

    def test_emissivity_metadata_rejection(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        # Create a satellite file without emissivity metadata
        sat_file = tmp_path / "thermal.npy"
        np.save(str(sat_file), np.random.uniform(290, 320, (50, 50)))

        with pytest.raises(IngestionValidationError) as exc_info:
            pipeline._ingest_satellite(sat_file)
        assert "emissivity" in str(exc_info.value).lower()
        assert exc_info.value.stream_name == "satellite"

    def test_material_classification_assigns_valid_properties(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        from src.shared.types import UDTVoxel

        voxels = [
            UDTVoxel(x=0, y=0, z=0, geometry_type="building",
                     albedo=0, emissivity=0, heat_capacity=0,
                     boundary_temp_k=300, material_class="concrete"),
            UDTVoxel(x=2, y=0, z=0, geometry_type="ground",
                     albedo=0, emissivity=0, heat_capacity=0,
                     boundary_temp_k=300, material_class="asphalt"),
        ]
        lib_path = tmp_path / "materials"
        lib_path.mkdir()
        pipeline._classify_materials(voxels, lib_path)

        for v in voxels:
            assert 0 <= v.albedo <= 1, f"Albedo out of range: {v.albedo}"
            assert 0 <= v.emissivity <= 1, f"Emissivity out of range: {v.emissivity}"
            assert v.heat_capacity > 0, f"Heat capacity must be positive: {v.heat_capacity}"

    def test_partial_stream_failure_continues(self, tmp_path):
        """Pipeline should log warning and continue when a non-critical stream fails."""
        pipeline = self._make_pipeline(tmp_path)
        # Ingest with nonexistent paths should still produce artifact
        artifact = pipeline.ingest(
            met_station_paths=[tmp_path / "nonexistent_station.npy"],
        )
        assert artifact.version_id is not None

    def test_kriging_interpolation_plausible_range(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        station_data = [
            {"station_id": "S1", "temperatures": np.full(288, 300.0),
             "location": np.array([10, 10])},
            {"station_id": "S2", "temperatures": np.full(288, 310.0),
             "location": np.array([190, 190])},
        ]
        resampled = pipeline._resample_to_hourly(station_data)
        interpolated = pipeline._kriging_interpolate(resampled)

        if interpolated is not None:
            assert interpolated.min() >= 290, "Interpolated temp too low"
            assert interpolated.max() <= 320, "Interpolated temp too high"

    def test_lidar_file_not_found_raises(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        with pytest.raises(IngestionValidationError) as exc_info:
            pipeline._ingest_lidar(tmp_path / "nonexistent.npy")
        assert exc_info.value.stream_name == "lidar"
