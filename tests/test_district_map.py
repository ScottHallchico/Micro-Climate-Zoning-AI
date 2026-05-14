"""Unit tests for Mumbai 3D District Map backend changes.

Covers:
- Zone classifier exhaustiveness (Requirement 4.2)
- Default fallback values for empty morphology rows (Requirement 4.5)
- GeoJSON metadata logging at API startup (Requirement 6.4)
- Missing GeoJSON file logs ERROR and health endpoint still returns 200 (Requirement 6.5)
- District_Block JSON serialisation round-trip (Requirement 4.4)
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers — import private functions from api module
# ---------------------------------------------------------------------------

import src.governance.api as api_module
from src.governance.api import (
    _as_float,
    _build_actual_blocks,
    _classify_actual_zone,
    _load_district_geometry_cache,
    app,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ZONE_CLASSES = {
    "WIND_CORRIDOR_CRITICAL",
    "THERMAL_REMEDIATION",
    "DENSITY_ADAPTIVE",
    "BASELINE_UNCHANGED",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def minimal_weather() -> dict[str, Any]:
    """Minimal weather dict sufficient for _build_actual_blocks."""
    return {
        "timestamp": "2024-01-15T12:00:00",
        "source": "test",
        "source_latitude": 19.07,
        "source_longitude": 72.87,
        "project_latitude": 19.07,
        "project_longitude": 72.87,
        "elevation_m": 14.0,
        "temperature_c": 32.0,
        "relative_humidity_pct": 65.0,
        "wind_speed_ms": 3.5,
        "wind_direction_deg": 270.0,
        "shortwave_radiation_wm2": 520.0,
        "surface_pressure_hpa": 1010.0,
        "soil_temperature_0cm_c": 35.0,
        "selection_mode": "design_peak",
    }


# ---------------------------------------------------------------------------
# 1. Zone classifier exhaustiveness (Requirement 4.2)
# ---------------------------------------------------------------------------


class TestZoneClassifierExhaustiveness:
    """Verify _classify_actual_zone always returns one of the four valid zone strings."""

    # Boundary values for each input dimension
    ROAD_COUNTS = [0, 14, 15, 30]          # below/at/above the roads >= 15 threshold
    GREEN_COVER_PCTS = [0.0, 7.0, 7.1, 20.0]  # below/at/above the green <= 7 threshold
    LAMBDA_P_VALUES = [0.0, 0.37, 0.38, 0.60]  # below/at/above the lambda_p >= 0.38 threshold
    MAX_HEIGHT_VALUES = [0.0, 17.9, 18.0, 40.0]  # below/at/above the height >= 18 threshold

    def _make_row(
        self,
        osm_road_count: int,
        green_cover_pct: float,
        lambda_p: float,
        max_height_m: float,
    ) -> dict[str, str]:
        return {
            "osm_road_count": str(osm_road_count),
            "green_cover_pct": str(green_cover_pct),
            "lambda_p": str(lambda_p),
            "max_height_m": str(max_height_m),
        }

    def test_all_boundary_combinations_return_valid_zone(self):
        """Exhaustive boundary-value test: all 4×4×4×4 = 256 combinations."""
        invalid_results: list[tuple] = []
        for roads in self.ROAD_COUNTS:
            for green in self.GREEN_COVER_PCTS:
                for lp in self.LAMBDA_P_VALUES:
                    for height in self.MAX_HEIGHT_VALUES:
                        row = self._make_row(roads, green, lp, height)
                        result = _classify_actual_zone(row)
                        if result not in VALID_ZONE_CLASSES:
                            invalid_results.append((roads, green, lp, height, result))

        assert not invalid_results, (
            f"_classify_actual_zone returned invalid zone class(es): {invalid_results}"
        )

    def test_high_road_count_returns_wind_corridor_critical(self):
        """roads >= 15 → WIND_CORRIDOR_CRITICAL regardless of other fields."""
        row = self._make_row(osm_road_count=15, green_cover_pct=20.0, lambda_p=0.1, max_height_m=5.0)
        assert _classify_actual_zone(row) == "WIND_CORRIDOR_CRITICAL"

    def test_low_green_high_lambda_returns_thermal_remediation(self):
        """green <= 7 AND lambda_p >= 0.38 → THERMAL_REMEDIATION (when roads < 15)."""
        row = self._make_row(osm_road_count=5, green_cover_pct=5.0, lambda_p=0.40, max_height_m=10.0)
        assert _classify_actual_zone(row) == "THERMAL_REMEDIATION"

    def test_tall_green_returns_density_adaptive(self):
        """height >= 18 AND green >= 10 → DENSITY_ADAPTIVE (when roads < 15, not thermal)."""
        row = self._make_row(osm_road_count=5, green_cover_pct=15.0, lambda_p=0.20, max_height_m=20.0)
        assert _classify_actual_zone(row) == "DENSITY_ADAPTIVE"

    def test_default_case_returns_baseline_unchanged(self):
        """No special conditions → BASELINE_UNCHANGED."""
        row = self._make_row(osm_road_count=5, green_cover_pct=15.0, lambda_p=0.20, max_height_m=10.0)
        assert _classify_actual_zone(row) == "BASELINE_UNCHANGED"

    def test_road_count_14_is_not_wind_corridor(self):
        """roads = 14 (below threshold) should NOT trigger WIND_CORRIDOR_CRITICAL."""
        row = self._make_row(osm_road_count=14, green_cover_pct=20.0, lambda_p=0.1, max_height_m=5.0)
        result = _classify_actual_zone(row)
        assert result != "WIND_CORRIDOR_CRITICAL"
        assert result in VALID_ZONE_CLASSES

    def test_green_exactly_7_triggers_thermal_remediation(self):
        """green = 7.0 (at threshold) with high lambda_p → THERMAL_REMEDIATION."""
        row = self._make_row(osm_road_count=5, green_cover_pct=7.0, lambda_p=0.38, max_height_m=10.0)
        assert _classify_actual_zone(row) == "THERMAL_REMEDIATION"

    def test_height_exactly_18_triggers_density_adaptive(self):
        """height = 18.0 (at threshold) with green >= 10 → DENSITY_ADAPTIVE."""
        row = self._make_row(osm_road_count=5, green_cover_pct=10.0, lambda_p=0.20, max_height_m=18.0)
        assert _classify_actual_zone(row) == "DENSITY_ADAPTIVE"

    def test_wind_corridor_takes_priority_over_thermal(self):
        """roads >= 15 takes priority even when thermal conditions are also met."""
        row = self._make_row(osm_road_count=20, green_cover_pct=5.0, lambda_p=0.50, max_height_m=5.0)
        assert _classify_actual_zone(row) == "WIND_CORRIDOR_CRITICAL"

    def test_wind_corridor_takes_priority_over_density_adaptive(self):
        """roads >= 15 takes priority even when density-adaptive conditions are also met."""
        row = self._make_row(osm_road_count=20, green_cover_pct=15.0, lambda_p=0.20, max_height_m=25.0)
        assert _classify_actual_zone(row) == "WIND_CORRIDOR_CRITICAL"


# ---------------------------------------------------------------------------
# 2. Default fallback values (Requirement 4.5)
# ---------------------------------------------------------------------------


class TestDefaultFallbackValues:
    """Verify all-empty morphology row produces BASELINE_UNCHANGED with correct defaults."""

    def _make_empty_row(self) -> dict[str, str]:
        """A district row where all morphology fields are empty strings."""
        return {
            "block_id": "test-district",
            "district_name": "",
            "col": "0",
            "row": "0",
            "osm_building_count": "",
            "osm_road_count": "",
            "osm_green_feature_count": "",
            "osm_residential_landuse_count": "",
            "measured_avg_building_levels": "",
            "max_height_m": "",
            "svf": "",
            "lambda_p": "",
            "hw_ratio": "",
            "albedo": "",
            "green_cover_pct": "",
            "centroid_lat": "",
            "centroid_lon": "",
            "area_km2": "",
        }

    def test_empty_row_classifies_as_baseline_unchanged(self):
        """All-empty morphology fields → BASELINE_UNCHANGED (Req 4.5)."""
        row = self._make_empty_row()
        result = _classify_actual_zone(row)
        assert result == "BASELINE_UNCHANGED"

    def test_empty_row_as_float_returns_zero_defaults(self):
        """_as_float on empty string returns 0.0 (the default)."""
        assert _as_float("") == 0.0
        assert _as_float(None) == 0.0

    def test_empty_row_as_float_custom_default(self):
        """_as_float respects a custom default value."""
        assert _as_float("", default=5.0) == 5.0
        assert _as_float(None, default=99.9) == 99.9

    def test_empty_morphology_fields_produce_zero_metrics(self):
        """Empty osm_road_count, green_cover_pct, lambda_p, max_height_m all parse to 0."""
        row = self._make_empty_row()
        assert int(_as_float(row.get("osm_road_count"))) == 0
        assert _as_float(row.get("green_cover_pct")) == 0.0
        assert _as_float(row.get("lambda_p")) == 0.0
        assert _as_float(row.get("max_height_m")) == 0.0

    def test_empty_row_does_not_trigger_wind_corridor(self):
        """roads = 0 (from empty) should not trigger WIND_CORRIDOR_CRITICAL."""
        row = self._make_empty_row()
        result = _classify_actual_zone(row)
        assert result != "WIND_CORRIDOR_CRITICAL"

    def test_empty_row_does_not_trigger_thermal_remediation(self):
        """green = 0 and lambda_p = 0 — lambda_p < 0.38 so no THERMAL_REMEDIATION."""
        row = self._make_empty_row()
        result = _classify_actual_zone(row)
        assert result != "THERMAL_REMEDIATION"

    def test_empty_row_does_not_trigger_density_adaptive(self):
        """height = 0 (< 18) so no DENSITY_ADAPTIVE."""
        row = self._make_empty_row()
        result = _classify_actual_zone(row)
        assert result != "DENSITY_ADAPTIVE"


# ---------------------------------------------------------------------------
# 3. GeoJSON metadata logging (Requirement 6.4)
# ---------------------------------------------------------------------------


class TestGeoJSONMetadataLogging:
    """Verify API startup logs metadata.source and metadata.retrieved_date at INFO level."""

    def _make_valid_geojson(self) -> dict:
        return {
            "type": "FeatureCollection",
            "metadata": {
                "source": "OpenStreetMap contributors",
                "license": "ODbL 1.0",
                "retrieved_date": "2024-01-15",
            },
            "features": [
                {
                    "type": "Feature",
                    "id": "test-district",
                    "properties": {"name": "Test District", "district": "Test District", "area_km2": 5.0},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[72.80, 18.90], [72.82, 18.90], [72.82, 18.92], [72.80, 18.92], [72.80, 18.90]]],
                    },
                }
            ],
        }

    def test_info_log_contains_metadata_source(self, caplog):
        """Startup should log metadata.source at INFO level."""
        geojson_data = self._make_valid_geojson()
        geojson_text = json.dumps(geojson_data)

        with caplog.at_level(logging.INFO, logger="src.governance.api"):
            with patch("src.governance.api.GEOJSON_PATH") as mock_path:
                mock_path.exists.return_value = True
                mock_path.open.return_value.__enter__ = lambda s: MagicMock(
                    read=lambda: geojson_text
                )
                # Use a real file-like object via io.StringIO
                import io
                mock_path.open.return_value = MagicMock()
                mock_path.open.return_value.__enter__ = lambda s: io.StringIO(geojson_text)
                mock_path.open.return_value.__exit__ = MagicMock(return_value=False)
                _load_district_geometry_cache()

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("OpenStreetMap contributors" in msg for msg in info_messages), (
            f"Expected metadata.source in INFO logs. Got: {info_messages}"
        )

    def test_info_log_contains_metadata_retrieved_date(self, caplog):
        """Startup should log metadata.retrieved_date at INFO level."""
        geojson_data = self._make_valid_geojson()
        geojson_text = json.dumps(geojson_data)

        with caplog.at_level(logging.INFO, logger="src.governance.api"):
            with patch("src.governance.api.GEOJSON_PATH") as mock_path:
                import io
                mock_path.exists.return_value = True
                mock_path.open.return_value = MagicMock()
                mock_path.open.return_value.__enter__ = lambda s: io.StringIO(geojson_text)
                mock_path.open.return_value.__exit__ = MagicMock(return_value=False)
                _load_district_geometry_cache()

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("2024-01-15" in msg for msg in info_messages), (
            f"Expected metadata.retrieved_date in INFO logs. Got: {info_messages}"
        )

    def test_info_log_contains_both_metadata_fields(self, caplog):
        """A single INFO log line should contain both source and retrieved_date."""
        geojson_data = self._make_valid_geojson()
        geojson_text = json.dumps(geojson_data)

        with caplog.at_level(logging.INFO, logger="src.governance.api"):
            with patch("src.governance.api.GEOJSON_PATH") as mock_path:
                import io
                mock_path.exists.return_value = True
                mock_path.open.return_value = MagicMock()
                mock_path.open.return_value.__enter__ = lambda s: io.StringIO(geojson_text)
                mock_path.open.return_value.__exit__ = MagicMock(return_value=False)
                _load_district_geometry_cache()

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        combined = " ".join(info_messages)
        assert "OpenStreetMap contributors" in combined
        assert "2024-01-15" in combined

    def test_cache_populated_after_successful_load(self):
        """After a successful load, _district_geometry_cache should have entries."""
        geojson_data = self._make_valid_geojson()
        geojson_text = json.dumps(geojson_data)

        with patch("src.governance.api.GEOJSON_PATH") as mock_path:
            import io
            mock_path.exists.return_value = True
            mock_path.open.return_value = MagicMock()
            mock_path.open.return_value.__enter__ = lambda s: io.StringIO(geojson_text)
            mock_path.open.return_value.__exit__ = MagicMock(return_value=False)
            _load_district_geometry_cache()

        assert "test-district" in api_module._district_geometry_cache
        geom = api_module._district_geometry_cache["test-district"]
        assert geom["type"] == "Polygon"


# ---------------------------------------------------------------------------
# 4. Missing GeoJSON at startup (Requirement 6.5)
# ---------------------------------------------------------------------------


class TestMissingGeoJSONAtStartup:
    """Verify missing GeoJSON logs ERROR and /v1/health still returns HTTP 200."""

    def test_missing_geojson_logs_error(self, caplog):
        """A missing GeoJSON file should log an ERROR-level message."""
        with caplog.at_level(logging.ERROR, logger="src.governance.api"):
            with patch("src.governance.api.GEOJSON_PATH") as mock_path:
                mock_path.exists.return_value = False
                mock_path.__str__ = lambda s: "/fake/path/mumbai_districts.geojson"
                _load_district_geometry_cache()

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert error_messages, "Expected at least one ERROR log when GeoJSON file is missing"

    def test_missing_geojson_error_mentions_file(self, caplog):
        """The ERROR log should reference the missing file path."""
        with caplog.at_level(logging.ERROR, logger="src.governance.api"):
            with patch("src.governance.api.GEOJSON_PATH") as mock_path:
                mock_path.exists.return_value = False
                mock_path.__str__ = lambda s: "/fake/path/mumbai_districts.geojson"
                _load_district_geometry_cache()

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        # The error should mention the path or "not found"
        combined = " ".join(error_messages)
        assert "not found" in combined.lower() or "geojson" in combined.lower() or "missing" in combined.lower(), (
            f"ERROR log should mention the missing file. Got: {error_messages}"
        )

    def test_health_endpoint_returns_200_when_geojson_missing(self):
        """Health endpoint must return HTTP 200 even when GeoJSON is absent (Req 6.5)."""
        client = TestClient(app)
        # The GeoJSON may or may not be present; health must always be 200
        with patch("src.governance.api.GEOJSON_PATH") as mock_path:
            mock_path.exists.return_value = False
            # Re-run cache load to simulate missing file scenario
            _load_district_geometry_cache()

        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_missing_geojson_does_not_populate_cache(self):
        """When GeoJSON is missing, the cache load is skipped (no new entries added).

        The function returns early without populating the cache, so any entries
        from a prior successful load are preserved (the cache is not reset on
        a missing-file error — it simply does not update).
        """
        # Reset the cache to a known empty state before the test
        original_cache = api_module._district_geometry_cache
        api_module._district_geometry_cache = {}
        try:
            with patch("src.governance.api.GEOJSON_PATH") as mock_path:
                mock_path.exists.return_value = False
                _load_district_geometry_cache()
            # Cache should remain empty — no entries were added
            assert api_module._district_geometry_cache == {}
        finally:
            # Restore original cache so other tests are not affected
            api_module._district_geometry_cache = original_cache

    def test_missing_metadata_source_logs_error(self, caplog):
        """Missing metadata.source field should log an ERROR."""
        geojson_data = {
            "type": "FeatureCollection",
            "metadata": {
                # "source" is intentionally missing
                "license": "ODbL 1.0",
                "retrieved_date": "2024-01-15",
            },
            "features": [],
        }
        geojson_text = json.dumps(geojson_data)

        with caplog.at_level(logging.ERROR, logger="src.governance.api"):
            with patch("src.governance.api.GEOJSON_PATH") as mock_path:
                import io
                mock_path.exists.return_value = True
                mock_path.open.return_value = MagicMock()
                mock_path.open.return_value.__enter__ = lambda s: io.StringIO(geojson_text)
                mock_path.open.return_value.__exit__ = MagicMock(return_value=False)
                _load_district_geometry_cache()

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert error_messages, "Expected ERROR log when metadata.source is missing"
        combined = " ".join(error_messages)
        assert "source" in combined.lower() or "metadata" in combined.lower()

    def test_missing_metadata_retrieved_date_logs_error(self, caplog):
        """Missing metadata.retrieved_date field should log an ERROR."""
        geojson_data = {
            "type": "FeatureCollection",
            "metadata": {
                "source": "OpenStreetMap contributors",
                "license": "ODbL 1.0",
                # "retrieved_date" is intentionally missing
            },
            "features": [],
        }
        geojson_text = json.dumps(geojson_data)

        with caplog.at_level(logging.ERROR, logger="src.governance.api"):
            with patch("src.governance.api.GEOJSON_PATH") as mock_path:
                import io
                mock_path.exists.return_value = True
                mock_path.open.return_value = MagicMock()
                mock_path.open.return_value.__enter__ = lambda s: io.StringIO(geojson_text)
                mock_path.open.return_value.__exit__ = MagicMock(return_value=False)
                _load_district_geometry_cache()

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert error_messages, "Expected ERROR log when metadata.retrieved_date is missing"
        combined = " ".join(error_messages)
        assert "retrieved_date" in combined.lower() or "metadata" in combined.lower()


# ---------------------------------------------------------------------------
# 5. District_Block serialisation (Requirement 4.4)
# ---------------------------------------------------------------------------


class TestDistrictBlockSerialisation:
    """Verify _build_actual_blocks output is JSON-serialisable and round-trips correctly."""

    def test_build_actual_blocks_returns_list(self, minimal_weather):
        """_build_actual_blocks should return a non-empty list."""
        blocks = _build_actual_blocks(minimal_weather)
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_all_blocks_are_json_serialisable(self, minimal_weather):
        """Every block dict must be serialisable to a JSON string without error."""
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            try:
                serialised = json.dumps(block)
            except (TypeError, ValueError) as exc:
                pytest.fail(
                    f"Block '{block.get('id', '?')}' is not JSON-serialisable: {exc}"
                )

    def test_json_round_trip_equality(self, minimal_weather):
        """json.loads(json.dumps(block)) == block for every block (Req 4.4)."""
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            serialised = json.dumps(block)
            deserialised = json.loads(serialised)
            assert deserialised == block, (
                f"Round-trip mismatch for block '{block.get('id', '?')}': "
                f"original={block}, round-tripped={deserialised}"
            )

    def test_blocks_contain_required_fields(self, minimal_weather):
        """Each block must contain all required District_Block fields (Req 4.1)."""
        required_fields = {
            "id", "district_name", "zone_class", "max_height_m", "uhi_intensity",
            "svf", "lambda_p", "hw_ratio", "albedo", "green_cover",
            "surface_temp_c", "air_temp_c", "centroid_lat", "centroid_lon",
            "area_km2", "col", "row",
        }
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            missing = required_fields - set(block.keys())
            assert not missing, (
                f"Block '{block.get('id', '?')}' is missing required fields: {missing}"
            )

    def test_zone_class_is_valid_for_all_blocks(self, minimal_weather):
        """Every block's zone_class must be one of the four valid strings (Req 4.2)."""
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            assert block["zone_class"] in VALID_ZONE_CLASSES, (
                f"Block '{block.get('id', '?')}' has invalid zone_class: {block['zone_class']}"
            )

    def test_geometry_field_is_none_or_dict(self, minimal_weather):
        """geometry field must be None or a dict (GeoJSON geometry object)."""
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            geom = block.get("geometry")
            assert geom is None or isinstance(geom, dict), (
                f"Block '{block.get('id', '?')}' has invalid geometry type: {type(geom)}"
            )

    def test_geometry_field_survives_json_round_trip(self, minimal_weather):
        """geometry coordinates must survive JSON serialisation without loss (Req 4.4)."""
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            geom = block.get("geometry")
            if geom is None:
                continue
            serialised = json.dumps(geom)
            deserialised = json.loads(serialised)
            assert deserialised == geom, (
                f"Geometry round-trip mismatch for block '{block.get('id', '?')}'"
            )

    def test_block_id_values_are_strings(self, minimal_weather):
        """All block id values must be non-empty strings."""
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            assert isinstance(block["id"], str) and block["id"], (
                f"Block has invalid id: {block.get('id')!r}"
            )

    def test_col_and_row_are_integers(self, minimal_weather):
        """col and row must be integers for legacy grid compatibility (Req 4.1)."""
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            assert isinstance(block["col"], int), (
                f"Block '{block['id']}' col is not int: {block['col']!r}"
            )
            assert isinstance(block["row"], int), (
                f"Block '{block['id']}' row is not int: {block['row']!r}"
            )

    def test_numeric_fields_are_finite_numbers(self, minimal_weather):
        """Key numeric fields must be finite floats/ints, not NaN or Inf."""
        import math
        numeric_fields = [
            "max_height_m", "uhi_intensity", "svf", "lambda_p",
            "hw_ratio", "albedo", "green_cover", "surface_temp_c", "air_temp_c",
        ]
        blocks = _build_actual_blocks(minimal_weather)
        for block in blocks:
            for field in numeric_fields:
                val = block.get(field)
                if val is not None:
                    assert isinstance(val, (int, float)), (
                        f"Block '{block['id']}' field '{field}' is not numeric: {val!r}"
                    )
                    assert math.isfinite(val), (
                        f"Block '{block['id']}' field '{field}' is not finite: {val}"
                    )
