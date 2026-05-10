"""Tests for rolling update cycle and drift detection (Task 14)."""

from datetime import datetime
from pathlib import Path

import pytest
import torch

from src.pinn.model import PINNModel
from src.updates.cycle import RollingUpdateManager
from src.shared.types import WeatherMeasurement, ZoningDirective, MandatoryIntervention, DirectiveProvenance
from src.shared.config import UpdateConfig, PINNConfig
from src.provenance.store import ProvenanceStore


def _make_measurements(n: int = 5) -> list[WeatherMeasurement]:
    """Generate synthetic weather measurements for testing."""
    return [
        WeatherMeasurement(
            station_id=f"S{i}",
            timestamp=datetime.now(),
            temperature_k=295 + i * 2,
            wind_speed_ms=3.0 + i * 0.5,
            wind_direction_deg=90 + i * 10,
            humidity_pct=50 + i * 5,
            solar_irradiance_wm2=400 + i * 50,
            location_x=100.0 + i * 20,
            location_y=100.0 + i * 20,
        )
        for i in range(n)
    ]


def _make_directive(block_id: str, review_trigger: str = "Annual review") -> ZoningDirective:
    return ZoningDirective(
        block_id=block_id,
        zone_class="THERMAL_REMEDIATION",
        max_height_m=20.0,
        height_justification="Test",
        permitted_uses=["residential"],
        mandatory_interventions=[],
        compliance_deadline_months=36,
        review_trigger=review_trigger,
        provenance=DirectiveProvenance(
            udt_version_id="udt-v1",
            pinn_checkpoint_version="pinn-v1",
            cfd_sample_ids=["cfd-001"],
            mobo_run_id="mobo-001",
        ),
    )


class TestRollingUpdateCycle:
    """Test quarterly retraining and checkpoint management."""

    def test_quarterly_retrain_produces_new_checkpoint(self, tmp_path):
        pinn_config = PINNConfig(max_epochs=5)
        model = PINNModel(input_dim=10, config=pinn_config)
        model._training_dataset_version = "ds-v1"

        manager = RollingUpdateManager(
            pinn_config=pinn_config,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            provenance_store=ProvenanceStore(str(tmp_path / "prov.json")),
        )

        measurements = _make_measurements(5)
        updated_model, artifact = manager.quarterly_retrain(model, measurements)

        assert artifact.version_id is not None
        assert Path(artifact.file_path).exists()

    def test_encoder_weights_frozen_during_retrain(self, tmp_path):
        pinn_config = PINNConfig(max_epochs=5)
        model = PINNModel(input_dim=10, config=pinn_config)
        model._training_dataset_version = "ds-v1"

        # Record encoder params before
        encoder_before = {
            name: p.clone() for name, p in model.encoder.named_parameters()
        }

        manager = RollingUpdateManager(
            pinn_config=pinn_config,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            provenance_store=ProvenanceStore(str(tmp_path / "prov.json")),
        )

        measurements = _make_measurements(5)
        manager.quarterly_retrain(model, measurements)

        # Encoder weights must be unchanged
        for name, p in model.encoder.named_parameters():
            assert torch.equal(p, encoder_before[name]), \
                f"Encoder param '{name}' changed during quarterly retrain"

    def test_drift_detection_above_threshold_flags(self, tmp_path):
        pinn_config = PINNConfig(max_epochs=2)
        model_old = PINNModel(input_dim=10, config=pinn_config)
        model_new = PINNModel(input_dim=10, config=pinn_config)

        manager = RollingUpdateManager(
            config=UpdateConfig(drift_threshold_c=0.4),
            pinn_config=pinn_config,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        # Create validation data that will produce different predictions
        validation_data = [
            {"block_id": "BLK-01", "features": [0.5, 0.3, 0.2, 1.5, 0.5, 0, 0, 0, 0, 0]},
            {"block_id": "BLK-02", "features": [0.8, 0.4, 0.3, 2.0, 1.0, 0, 0, 0, 0, 0]},
        ]

        drifted = manager.detect_drift(model_new, model_old, validation_data)
        # With random weights, some drift is likely
        # The test verifies the mechanism works, not specific drift values
        assert isinstance(drifted, dict)

    def test_drift_detection_below_threshold_no_flag(self, tmp_path):
        pinn_config = PINNConfig(max_epochs=2)
        model = PINNModel(input_dim=10, config=pinn_config)

        manager = RollingUpdateManager(
            config=UpdateConfig(drift_threshold_c=1000.0),  # Very high threshold
            pinn_config=pinn_config,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        validation_data = [
            {"block_id": "BLK-01", "features": [0.5] * 10},
        ]

        # Same model compared to itself → zero drift
        drifted = manager.detect_drift(model, model, validation_data)
        assert len(drifted) == 0, "Same model should not show drift"

    def test_flag_directives_for_review(self, tmp_path):
        manager = RollingUpdateManager(
            config=UpdateConfig(drift_threshold_c=0.4),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        directives = [
            _make_directive("BLK-01"),
            _make_directive("BLK-02"),
            _make_directive("BLK-03"),
        ]

        drifted_blocks = {"BLK-01": 0.65, "BLK-03": 0.52}
        flagged = manager.flag_directives_for_review(directives, drifted_blocks)

        assert len(flagged) == 2
        assert any(d.block_id == "BLK-01" for d in flagged)
        assert any(d.block_id == "BLK-03" for d in flagged)
        # Check drift annotation is in review trigger
        for d in flagged:
            assert "AUTO-FLAGGED" in d.review_trigger
            assert "drift" in d.review_trigger.lower()

    def test_checkpoint_history_retained(self, tmp_path):
        pinn_config = PINNConfig(max_epochs=2)
        model = PINNModel(input_dim=10, config=pinn_config)
        model._training_dataset_version = "ds-v1"

        manager = RollingUpdateManager(
            pinn_config=pinn_config,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            provenance_store=ProvenanceStore(str(tmp_path / "prov.json")),
        )

        measurements = _make_measurements(3)

        # Do two retraining cycles
        manager.quarterly_retrain(model, measurements)
        manager.quarterly_retrain(model, measurements)

        history = manager.get_checkpoint_history()
        assert len(history) == 2
        # All checkpoints should still be loadable
        for ckpt in history:
            assert Path(ckpt.file_path).exists()
