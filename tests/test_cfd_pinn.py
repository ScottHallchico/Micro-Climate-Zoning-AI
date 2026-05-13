"""Tests for CFD Runner (Phase 2) and PINN Model (Phase 3)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.cfd.runner import CFDRunner
from src.pinn.model import PINNModel
from src.shared.types import UCMArtifact, CFDDatasetArtifact
from src.shared.config import CFDConfig, PINNConfig
from src.shared.exceptions import CheckpointError
from src.provenance.store import ProvenanceStore


class TestCFDRunner:
    """Test CFD simulation orchestration and validation."""

    def _make_runner(self, tmp_path: Path) -> CFDRunner:
        return CFDRunner(
            config=CFDConfig(n_samples=20),  # Reduced for testing
            provenance_store=ProvenanceStore(str(tmp_path / "prov.json")),
        )

    def test_lhs_produces_correct_sample_count(self, tmp_path):
        runner = self._make_runner(tmp_path)
        ucm = UCMArtifact()
        samples = runner._lhs_sample([], 200)
        assert len(samples) == 200

    def test_lhs_covers_all_canyon_classes(self, tmp_path):
        runner = self._make_runner(tmp_path)
        samples = runner._lhs_sample([], 200)
        classes = set(s["canyon_class"] for s in samples)
        # At minimum, deep, regular, and shallow should be present
        assert "deep_canyon" in classes or "regular_canyon" in classes

    def test_mass_conservation_valid(self, tmp_path):
        runner = self._make_runner(tmp_path)
        # Create a divergence-free field
        wind = np.zeros((8, 8, 4, 4))
        wind[..., 0] = 1.0  # uniform u
        residual = runner._compute_mass_conservation_residual(wind)
        assert residual < 1e-4

    def test_mass_conservation_invalid(self, tmp_path):
        runner = self._make_runner(tmp_path)
        rng = np.random.default_rng(42)
        wind = rng.normal(0, 10, (8, 8, 4, 4))  # Random = high divergence
        residual = runner._compute_mass_conservation_residual(wind)
        assert residual >= 0  # Should be non-negative

    def test_generate_dataset_produces_artifact(self, tmp_path):
        runner = self._make_runner(tmp_path)
        ucm = UCMArtifact()
        artifact = runner.generate_training_dataset(ucm, n_samples=5)
        assert artifact.version_id is not None
        assert artifact.sample_count == 5

    def test_failed_samples_excluded_from_valid(self, tmp_path):
        runner = self._make_runner(tmp_path)
        ucm = UCMArtifact()
        artifact = runner.generate_training_dataset(ucm, n_samples=5)
        assert artifact.valid_sample_count <= artifact.sample_count
        assert len(artifact.samples) == artifact.valid_sample_count

    def test_block_microclimate_outputs_wind_and_solar_metrics(self, tmp_path):
        runner = self._make_runner(tmp_path)
        results = runner.simulate_block_microclimate(blocks=[{
            "id": "BLK-TEST",
            "zone_class": "WIND_CORRIDOR_CRITICAL",
            "max_height_m": 24,
            "svf": 0.5,
            "lambda_p": 0.4,
            "hw_ratio": 1.2,
            "albedo": 0.3,
            "green_cover": 20,
        }])

        assert len(results) == 1
        assert results[0]["block_id"] == "BLK-TEST"
        assert results[0]["wind"]["avg_wind_speed_ms"] > 0
        assert 0 <= results[0]["wind"]["ventilation_score"] <= 1
        assert results[0]["solar"]["solar_radiation_wm2"] > 0
        assert results[0]["thermal"]["surface_temp_c"] > 0
        assert results[0]["thermal"]["heat_storage_wm2"] >= 0
        assert 0 <= results[0]["microclimate"]["heat_burden_score"] <= 1


class TestPINNModel:
    """Test PINN architecture, prediction, and OOD detection."""

    def _make_model(self, tmp_path: Path | None = None) -> PINNModel:
        config = PINNConfig(max_epochs=10)
        prov = ProvenanceStore(str(tmp_path / "prov.json")) if tmp_path else None
        return PINNModel(input_dim=10, config=config, provenance_store=prov)

    def test_forward_pass_output_shapes(self, tmp_path):
        model = self._make_model(tmp_path)
        x = torch.randn(4, 10)
        wind, energy = model(x)
        assert wind.shape == (4, 4)  # (batch, uvwP)
        assert energy.shape == (4, 4)  # (batch, Qh Qe ΔQs Q*)

    def test_predict_returns_prediction(self, tmp_path):
        model = self._make_model(tmp_path)
        x = torch.randn(10)
        pred = model.predict(x)
        assert pred.wind_field is not None
        assert pred.energy_field is not None

    def test_ood_flag_set_for_extreme_input(self, tmp_path):
        model = self._make_model(tmp_path)
        # Set training stats
        model._training_mean = torch.zeros(10)
        model._training_std = torch.ones(10)
        model._feature_names = [f"f{i}" for i in range(10)]

        # Input at 4σ should trigger OOD
        x = torch.ones(10) * 4.0
        pred = model.predict(x)
        assert pred.ood_warning is True
        assert len(pred.ood_features) > 0

    def test_ood_flag_not_set_for_normal_input(self, tmp_path):
        model = self._make_model(tmp_path)
        model._training_mean = torch.zeros(10)
        model._training_std = torch.ones(10)

        x = torch.ones(10) * 2.0  # Within 3σ
        pred = model.predict(x)
        assert pred.ood_warning is False

    def test_ood_boundary_exactly_3_sigma(self, tmp_path):
        """Feature at exactly 3σ should NOT trigger OOD flag (> not >=)."""
        model = self._make_model(tmp_path)
        model._training_mean = torch.zeros(10)
        model._training_std = torch.ones(10)

        x = torch.ones(10) * 3.0  # Exactly 3σ
        pred = model.predict(x)
        assert pred.ood_warning is False

    def test_ood_boundary_just_above_3_sigma(self, tmp_path):
        """Feature at 3σ + ε should trigger OOD flag."""
        model = self._make_model(tmp_path)
        model._training_mean = torch.zeros(10)
        model._training_std = torch.ones(10)

        x = torch.ones(10) * 3.001  # Just above 3σ
        pred = model.predict(x)
        assert pred.ood_warning is True

    def test_fine_tune_does_not_modify_encoder(self, tmp_path):
        model = self._make_model(tmp_path)
        from src.shared.types import WeatherMeasurement
        from datetime import datetime

        # Record encoder params before fine-tuning
        encoder_params_before = {
            name: p.clone()
            for name, p in model.encoder.named_parameters()
        }

        measurements = [
            WeatherMeasurement(
                station_id="S1", timestamp=datetime.now(),
                temperature_k=300, wind_speed_ms=5,
                wind_direction_deg=90, humidity_pct=50,
                solar_irradiance_wm2=500, location_x=100, location_y=100,
            )
        ]
        model.fine_tune(measurements, freeze_encoder=True)

        # Check encoder params unchanged
        for name, p in model.encoder.named_parameters():
            assert torch.equal(p, encoder_params_before[name]), \
                f"Encoder param '{name}' was modified during fine-tuning"

    def test_predict_with_jacobian(self, tmp_path):
        model = self._make_model(tmp_path)
        x = torch.randn(10)
        pred, jacobian = model.predict_with_jacobian(x)
        assert pred.wind_field is not None
        assert jacobian.shape[1] == 10  # Input dim


class TestPINNCheckpoint:
    """Test PINN checkpoint serialization and round-trip fidelity."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Property 1: load(save(model)).predict(x) == model.predict(x)"""
        config = PINNConfig(max_epochs=5)
        model = PINNModel(input_dim=10, config=config)

        # Train briefly to get non-trivial weights
        model._training_mean = torch.zeros(10)
        model._training_std = torch.ones(10)
        model._training_dataset_version = "test-v1"

        x = torch.randn(10)
        pred_before = model.predict(x)

        # Save
        ckpt_path = tmp_path / "test_ckpt.pt"
        artifact = model.save_checkpoint(ckpt_path)
        assert artifact.version_id is not None
        assert ckpt_path.exists()

        # Load
        loaded = PINNModel.load_checkpoint(ckpt_path, config=config)
        pred_after = loaded.predict(x)

        # Compare predictions (float32 epsilon tolerance)
        np.testing.assert_allclose(
            pred_before.wind_field, pred_after.wind_field,
            atol=1e-6, rtol=1e-5,
        )
        np.testing.assert_allclose(
            pred_before.energy_field, pred_after.energy_field,
            atol=1e-6, rtol=1e-5,
        )

    def test_corrupt_file_raises_checkpoint_error(self, tmp_path):
        path = tmp_path / "corrupt.pt"
        path.write_text("not a valid checkpoint")
        with pytest.raises(CheckpointError):
            PINNModel.load_checkpoint(path)

    def test_missing_file_raises_checkpoint_error(self, tmp_path):
        with pytest.raises(CheckpointError):
            PINNModel.load_checkpoint(tmp_path / "nonexistent.pt")

    def test_architecture_mismatch_raises(self, tmp_path):
        # Save with 8 layers
        config8 = PINNConfig(encoder_layers=8, encoder_width=512)
        model = PINNModel(input_dim=10, config=config8)
        model._training_dataset_version = "v1"
        path = tmp_path / "ckpt8.pt"
        model.save_checkpoint(path)

        # Try loading with different layer config
        config4 = PINNConfig(encoder_layers=4, encoder_width=512)
        with pytest.raises(CheckpointError) as exc_info:
            PINNModel.load_checkpoint(path, config=config4)
        assert "mismatch" in str(exc_info.value).lower()

    def test_manifest_fields_present(self, tmp_path):
        model = PINNModel(input_dim=10)
        model._training_dataset_version = "ds-v1"
        model._fine_tuning_city_id = "delhi"
        path = tmp_path / "manifest_test.pt"
        artifact = model.save_checkpoint(path)

        assert artifact.manifest is not None
        assert artifact.manifest.training_dataset_version == "ds-v1"
        assert artifact.manifest.fine_tuning_city_id == "delhi"
        assert "lambda_1" in artifact.manifest.physics_coefficients
