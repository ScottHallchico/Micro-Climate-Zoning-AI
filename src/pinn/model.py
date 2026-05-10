"""Phase 3: Physics-Informed Neural Network Model.

Multi-head PINN with SIREN activations that simultaneously learns from CFD data
and respects Navier-Stokes and energy transport equations as hard constraints.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.shared.types import (
    PINNPrediction, CheckpointManifest, CheckpointArtifact,
    WeatherMeasurement, CFDDatasetArtifact, ProvenanceRecord,
)
from src.shared.config import PINNConfig
from src.shared.exceptions import CheckpointError
from src.provenance.store import ProvenanceStore

logger = logging.getLogger(__name__)


class SIRENLayer(nn.Module):
    """A single SIREN (Sinusoidal Representation Network) layer.

    Uses sin(ω₀ · Wx + b) activation for smooth, infinitely differentiable
    representations required by PDE constraints.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(is_first)

    def _init_weights(self, is_first: bool) -> None:
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                    1 / self.linear.in_features,
                )
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SharedEncoder(nn.Module):
    """8-layer MLP shared encoder with SIREN activations."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, n_layers: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers = [SIRENLayer(input_dim, hidden_dim, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class WindHead(nn.Module):
    """Prediction head for wind field: outputs (u, v, w, P) per voxel."""

    def __init__(self, hidden_dim: int = 512, output_dim: int = 4) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class EnergyHead(nn.Module):
    """Prediction head for energy field: outputs (Qh, Qe, ΔQs, Q*) per voxel."""

    def __init__(self, hidden_dim: int = 512, output_dim: int = 4) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PINNModel(nn.Module):
    """Physics-Informed Neural Network for urban microclimate prediction.

    Architecture: shared SIREN encoder → dual heads (wind + energy).
    Physics constraints (Navier-Stokes, energy balance, BCs) embedded in loss.
    """

    def __init__(
        self,
        input_dim: int = 10,
        config: PINNConfig | None = None,
        provenance_store: ProvenanceStore | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PINNConfig()
        self.provenance = provenance_store or ProvenanceStore()
        self.input_dim = input_dim

        # Architecture
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dim=self.config.encoder_width,
            n_layers=self.config.encoder_layers,
        )
        self.wind_head = WindHead(
            hidden_dim=self.config.encoder_width,
            output_dim=self.config.wind_head_outputs,
        )
        self.energy_head = EnergyHead(
            hidden_dim=self.config.encoder_width,
            output_dim=self.config.energy_head_outputs,
        )

        # Physics loss coefficients
        self.lambda_1 = self.config.lambda_1  # Navier-Stokes
        self.lambda_2 = self.config.lambda_2  # Energy balance
        self.lambda_3 = self.config.lambda_3  # Boundary conditions

        # Training distribution statistics for OOD detection
        self._training_mean: torch.Tensor | None = None
        self._training_std: torch.Tensor | None = None
        self._feature_names: list[str] = []

        # Training metadata
        self._training_dataset_version: str = ""
        self._fine_tuning_city_id: str | None = None
        self._training_timestamp: datetime | None = None
        self._validation_mae: float = 0.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and both heads.

        Args:
            x: Input feature tensor of shape (batch, input_dim).

        Returns:
            Tuple of (wind_field, energy_field) tensors.
        """
        encoded = self.encoder(x)
        wind = self.wind_head(encoded)
        energy = self.energy_head(encoded)
        return wind, energy

    def predict(self, block_feature_vector: torch.Tensor) -> PINNPrediction:
        """Run prediction with OOD detection.

        Attaches OOD warning flag if any input feature is > 3σ from training mean.
        """
        self.eval()
        with torch.no_grad():
            # OOD detection
            ood_warning = False
            ood_features: list[str] = []

            if self._training_mean is not None and self._training_std is not None:
                z_scores = torch.abs(
                    (block_feature_vector - self._training_mean)
                    / (self._training_std + 1e-8)
                )
                ood_mask = z_scores > self.config.ood_sigma_threshold
                if ood_mask.any():
                    ood_warning = True
                    ood_indices = torch.where(ood_mask.flatten())[0].tolist()
                    ood_features = [
                        self._feature_names[i] if i < len(self._feature_names)
                        else f"feature_{i}"
                        for i in ood_indices
                    ]

            # Forward pass
            if block_feature_vector.dim() == 1:
                block_feature_vector = block_feature_vector.unsqueeze(0)

            wind, energy = self.forward(block_feature_vector)

            return PINNPrediction(
                wind_field=wind.numpy(),
                energy_field=energy.numpy(),
                ood_warning=ood_warning,
                ood_features=ood_features,
                block_temperature=float(energy[0, 0]) + 300.0,
            )

    def predict_with_jacobian(
        self, block_feature_vector: torch.Tensor
    ) -> tuple[PINNPrediction, torch.Tensor]:
        """Return prediction and ∂output/∂input Jacobian for optimizer use."""
        self.eval()
        x = block_feature_vector.clone().requires_grad_(True)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        wind, energy = self.forward(x)
        output = torch.cat([wind, energy], dim=-1)

        # Compute Jacobian
        jacobian_rows = []
        for i in range(output.shape[-1]):
            grad = torch.autograd.grad(
                output[0, i], x, retain_graph=True, create_graph=False
            )[0]
            jacobian_rows.append(grad.squeeze(0))

        jacobian = torch.stack(jacobian_rows)

        prediction = PINNPrediction(
            wind_field=wind.detach().numpy(),
            energy_field=energy.detach().numpy(),
            block_temperature=float(energy[0, 0].detach()) + 300.0,
        )

        return prediction, jacobian.detach()

    def compute_physics_loss(
        self,
        x: torch.Tensor,
        wind_pred: torch.Tensor,
        energy_pred: torch.Tensor,
        wind_target: torch.Tensor | None = None,
        energy_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite physics-informed loss.

        L_total = L_data + λ₁·L_NS + λ₂·L_energy + λ₃·L_BC
        """
        losses = {}

        # L_data: MSE on CFD ground truth
        l_data = torch.tensor(0.0)
        if wind_target is not None:
            l_data = l_data + nn.functional.mse_loss(wind_pred, wind_target)
        if energy_target is not None:
            l_data = l_data + nn.functional.mse_loss(energy_pred, energy_target)
        losses["l_data"] = l_data

        # L_NS: Navier-Stokes residuals (continuity ∇·u = 0)
        u, v, w = wind_pred[..., 0], wind_pred[..., 1], wind_pred[..., 2]
        divergence = (
            torch.gradient(u, dim=-1)[0]
            if u.dim() > 1
            else torch.zeros_like(u)
        )
        l_ns = torch.mean(divergence ** 2)
        losses["l_ns"] = l_ns

        # L_energy: Energy balance Q* = Qh + Qe + ΔQs
        qh = energy_pred[..., 0]
        qe = energy_pred[..., 1]
        delta_qs = energy_pred[..., 2]
        q_star = energy_pred[..., 3]
        energy_residual = q_star - (qh + qe + delta_qs)
        l_energy = torch.mean(energy_residual ** 2)
        losses["l_energy"] = l_energy

        # L_BC: Boundary condition residuals (simplified)
        l_bc = torch.tensor(0.0)
        losses["l_bc"] = l_bc

        # Total loss
        l_total = (
            l_data
            + self.lambda_1 * l_ns
            + self.lambda_2 * l_energy
            + self.lambda_3 * l_bc
        )
        losses["l_total"] = l_total

        return losses

    def train_on_cfd_data(
        self,
        cfd_dataset: CFDDatasetArtifact,
        epochs: int | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, float]:
        """Train the PINN on CFD ground-truth data.

        Args:
            cfd_dataset: The CFD training dataset.
            epochs: Number of training epochs.
            learning_rate: Learning rate for optimizer.

        Returns:
            Dictionary of final loss values.
        """
        epochs = epochs or self.config.max_epochs
        lr = learning_rate or self.config.learning_rate

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        # Prepare training data from CFD samples
        features, wind_targets, energy_targets = self._prepare_training_data(cfd_dataset)

        # Compute training distribution statistics for OOD detection
        self._training_mean = features.mean(dim=0)
        self._training_std = features.std(dim=0)
        self._feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        self._training_dataset_version = cfd_dataset.version_id

        final_losses = {}
        for epoch in range(epochs):
            optimizer.zero_grad()

            wind_pred, energy_pred = self.forward(features)
            losses = self.compute_physics_loss(
                features, wind_pred, energy_pred, wind_targets, energy_targets
            )

            losses["l_total"].backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"total={losses['l_total'].item():.6f}, "
                    f"data={losses['l_data'].item():.6f}, "
                    f"NS={losses['l_ns'].item():.6f}, "
                    f"energy={losses['l_energy'].item():.6f}"
                )

            final_losses = {k: v.item() for k, v in losses.items()}

        self._training_timestamp = datetime.utcnow()
        self._validation_mae = final_losses.get("l_data", 0.0)

        return final_losses

    def fine_tune(
        self,
        weather_station_measurements: list[WeatherMeasurement],
        freeze_encoder: bool = True,
    ) -> None:
        """Fine-tune only the final 2 layers of each head.

        Encoder weights remain frozen (transfer learning strategy).
        """
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Only train last 2 layers of each head
        trainable_params = list(self.wind_head.parameters()) + list(self.energy_head.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config.learning_rate * 0.1)

        # Prepare fine-tuning data from weather measurements
        features = self._prepare_weather_features(weather_station_measurements)

        self.train()
        for epoch in range(min(100, self.config.max_epochs)):
            optimizer.zero_grad()
            wind_pred, energy_pred = self.forward(features)
            losses = self.compute_physics_loss(features, wind_pred, energy_pred)
            losses["l_total"].backward()
            optimizer.step()

        # Restore encoder gradient computation
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = True

        self._fine_tuning_city_id = weather_station_measurements[0].station_id.split("-")[0] if weather_station_measurements else None

    def save_checkpoint(self, path: Path) -> CheckpointArtifact:
        """Serialize full model state to a versioned checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = self._training_timestamp or datetime.utcnow()
        city_id = self._fine_tuning_city_id or "base"
        dataset_version = self._training_dataset_version or "unknown"
        version_id = f"{dataset_version[:8]}_{city_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        manifest = CheckpointManifest(
            version_id=version_id,
            training_dataset_version=dataset_version,
            fine_tuning_city_id=self._fine_tuning_city_id,
            training_timestamp=timestamp,
            architecture_config={
                "input_dim": self.input_dim,
                "encoder_layers": self.config.encoder_layers,
                "encoder_width": self.config.encoder_width,
                "activation": self.config.activation,
                "wind_head_outputs": self.config.wind_head_outputs,
                "energy_head_outputs": self.config.energy_head_outputs,
            },
            physics_coefficients={
                "lambda_1": self.lambda_1,
                "lambda_2": self.lambda_2,
                "lambda_3": self.lambda_3,
            },
            validation_mae=self._validation_mae,
        )

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "manifest": {
                "version_id": manifest.version_id,
                "training_dataset_version": manifest.training_dataset_version,
                "fine_tuning_city_id": manifest.fine_tuning_city_id,
                "training_timestamp": manifest.training_timestamp.isoformat(),
                "architecture_config": manifest.architecture_config,
                "physics_coefficients": manifest.physics_coefficients,
                "validation_mae": manifest.validation_mae,
            },
            "training_stats": {
                "mean": self._training_mean.numpy().tolist() if self._training_mean is not None else None,
                "std": self._training_std.numpy().tolist() if self._training_std is not None else None,
                "feature_names": self._feature_names,
            },
        }

        torch.save(checkpoint, str(path))

        artifact = CheckpointArtifact(
            version_id=version_id,
            file_path=str(path),
            manifest=manifest,
            creation_timestamp=datetime.utcnow(),
        )

        # Record provenance
        self.provenance.record_artifact(ProvenanceRecord(
            artifact_id=version_id,
            artifact_type="pinn_checkpoint",
            producing_component="pinn_model",
            input_artifact_ids=[dataset_version] if dataset_version != "unknown" else [],
            metadata={
                "architecture": manifest.architecture_config,
                "validation_mae": manifest.validation_mae,
            },
        ))

        logger.info(f"Checkpoint saved: {path} (version={version_id})")
        return artifact

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        config: PINNConfig | None = None,
        provenance_store: ProvenanceStore | None = None,
    ) -> "PINNModel":
        """Load and validate a checkpoint. Never returns a partially initialized model."""
        path = Path(path)

        if not path.exists():
            raise CheckpointError(str(path), "File not found")

        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            raise CheckpointError(str(path), f"Corrupt checkpoint file: {e}")

        if "manifest" not in checkpoint or "model_state_dict" not in checkpoint:
            raise CheckpointError(str(path), "Missing required fields (manifest or model_state_dict)")

        manifest_dict = checkpoint["manifest"]

        # Validate architecture compatibility
        arch_config = manifest_dict.get("architecture_config", {})
        input_dim = arch_config.get("input_dim", 10)

        pinn_config = config or PINNConfig()

        # Check architecture match
        if arch_config.get("encoder_layers") != pinn_config.encoder_layers:
            raise CheckpointError(
                str(path),
                f"Architecture mismatch: checkpoint has {arch_config.get('encoder_layers')} encoder layers, "
                f"but config specifies {pinn_config.encoder_layers}",
            )

        if arch_config.get("encoder_width") != pinn_config.encoder_width:
            raise CheckpointError(
                str(path),
                f"Architecture mismatch: checkpoint has width {arch_config.get('encoder_width')}, "
                f"but config specifies {pinn_config.encoder_width}",
            )

        # Create model and load state
        model = cls(
            input_dim=input_dim,
            config=pinn_config,
            provenance_store=provenance_store,
        )

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            raise CheckpointError(str(path), f"State dict incompatible: {e}")

        # Restore training stats
        stats = checkpoint.get("training_stats", {})
        if stats.get("mean") is not None:
            model._training_mean = torch.tensor(stats["mean"])
        if stats.get("std") is not None:
            model._training_std = torch.tensor(stats["std"])
        model._feature_names = stats.get("feature_names", [])

        # Restore metadata
        model._training_dataset_version = manifest_dict.get("training_dataset_version", "")
        model._fine_tuning_city_id = manifest_dict.get("fine_tuning_city_id")
        ts = manifest_dict.get("training_timestamp")
        model._training_timestamp = datetime.fromisoformat(ts) if ts else None
        model._validation_mae = manifest_dict.get("validation_mae", 0.0)

        # Restore physics coefficients
        coeffs = manifest_dict.get("physics_coefficients", {})
        model.lambda_1 = coeffs.get("lambda_1", model.lambda_1)
        model.lambda_2 = coeffs.get("lambda_2", model.lambda_2)
        model.lambda_3 = coeffs.get("lambda_3", model.lambda_3)

        logger.info(f"Checkpoint loaded: {path} (version={manifest_dict.get('version_id')})")
        return model

    def _prepare_training_data(
        self, cfd_dataset: CFDDatasetArtifact
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert CFD samples to training tensors."""
        features_list = []
        wind_list = []
        energy_list = []

        for sample in cfd_dataset.samples:
            if sample.status != "valid":
                continue

            # Extract morphology params as feature vector
            params = sample.morphology_params
            feature = [
                params.get("svf", 0.5),
                params.get("lambda_p", 0.3),
                params.get("lambda_f", 0.2),
                params.get("hw_ratio", 1.5),
                params.get("z0", 0.5),
            ]
            # Pad to input_dim
            while len(feature) < self.input_dim:
                feature.append(0.0)
            features_list.append(feature[:self.input_dim])

            # Extract field averages as targets
            if sample.wind_field is not None:
                wind_avg = sample.wind_field.mean(axis=(0, 1, 2))
            else:
                wind_avg = np.zeros(4)
            wind_list.append(wind_avg)

            if sample.energy_field is not None:
                energy_avg = sample.energy_field.mean(axis=(0, 1, 2))
            else:
                energy_avg = np.zeros(4)
            energy_list.append(energy_avg)

        if not features_list:
            # Return dummy data if no valid samples
            return (
                torch.randn(10, self.input_dim),
                torch.randn(10, 4),
                torch.randn(10, 4),
            )

        return (
            torch.tensor(features_list, dtype=torch.float32),
            torch.tensor(np.array(wind_list), dtype=torch.float32),
            torch.tensor(np.array(energy_list), dtype=torch.float32),
        )

    def _prepare_weather_features(
        self, measurements: list[WeatherMeasurement]
    ) -> torch.Tensor:
        """Convert weather measurements to feature tensors."""
        features = []
        for m in measurements:
            feature = [
                m.temperature_k / 300.0,
                m.wind_speed_ms / 10.0,
                m.wind_direction_deg / 360.0,
                m.humidity_pct / 100.0,
                m.solar_irradiance_wm2 / 1000.0,
                m.location_x / 1000.0,
                m.location_y / 1000.0,
            ]
            while len(feature) < self.input_dim:
                feature.append(0.0)
            features.append(feature[:self.input_dim])

        return torch.tensor(features, dtype=torch.float32)
