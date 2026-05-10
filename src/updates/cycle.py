"""Rolling model update cycle — quarterly PINN retraining with drift detection.

Ingests new AWS meteorological data and satellite thermal imagery, fine-tunes the PINN,
detects model drift, and auto-flags zoning directives for review when drift exceeds 0.4°C.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.shared.types import (
    WeatherMeasurement, CheckpointArtifact, ZoningDirective, ProvenanceRecord,
)
from src.shared.config import UpdateConfig, PINNConfig
from src.pinn.model import PINNModel
from src.provenance.store import ProvenanceStore

logger = logging.getLogger(__name__)


class RollingUpdateManager:
    """Manages quarterly PINN retraining and drift detection.

    The PINN is retrained quarterly with new sensor data. If model drift
    exceeds 0.4°C on any block, all active directives for that block are
    auto-flagged for human review.
    """

    def __init__(
        self,
        config: UpdateConfig | None = None,
        pinn_config: PINNConfig | None = None,
        provenance_store: ProvenanceStore | None = None,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        self.config = config or UpdateConfig()
        self.pinn_config = pinn_config or PINNConfig()
        self.provenance = provenance_store or ProvenanceStore()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_history: list[CheckpointArtifact] = []

    def quarterly_retrain(
        self,
        current_model: PINNModel,
        new_measurements: list[WeatherMeasurement],
        validation_data: list[dict[str, Any]] | None = None,
    ) -> tuple[PINNModel, CheckpointArtifact]:
        """Run quarterly retraining cycle.

        Fine-tunes the PINN on new measurements without full retraining.
        Saves a new versioned checkpoint and retains all previous checkpoints.

        Args:
            current_model: The current deployed PINN model.
            new_measurements: New weather station measurements since last cycle.
            validation_data: Optional held-out validation data for drift detection.

        Returns:
            Tuple of (updated model, new checkpoint artifact).
        """
        logger.info(f"Starting quarterly retraining with {len(new_measurements)} new measurements")

        # Fine-tune only final layers (encoder stays frozen)
        current_model.fine_tune(
            weather_station_measurements=new_measurements,
            freeze_encoder=True,
        )

        # Save new checkpoint
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"pinn_quarterly_{timestamp}.pt"
        artifact = current_model.save_checkpoint(checkpoint_path)

        self._checkpoint_history.append(artifact)

        logger.info(f"Quarterly retraining complete. Checkpoint: {checkpoint_path}")
        return current_model, artifact

    def detect_drift(
        self,
        new_model: PINNModel,
        previous_model: PINNModel,
        validation_data: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Detect model drift by comparing predictions on validation set.

        Computes per-block MAE between new and previous model predictions.

        Args:
            new_model: The newly retrained model.
            previous_model: The previous model version.
            validation_data: List of validation samples with block_id and features.

        Returns:
            Dict of block_id -> MAE (°C) for blocks exceeding drift threshold.
        """
        drifted_blocks: dict[str, float] = {}

        for sample in validation_data:
            block_id = sample.get("block_id", "unknown")
            features = torch.tensor(
                sample.get("features", [0.0] * new_model.input_dim),
                dtype=torch.float32,
            )

            new_pred = new_model.predict(features)
            old_pred = previous_model.predict(features)

            mae = abs(new_pred.block_temperature - old_pred.block_temperature)

            if mae > self.config.drift_threshold_c:
                drifted_blocks[block_id] = float(mae)
                logger.warning(
                    f"Drift detected on block {block_id}: "
                    f"MAE={mae:.2f}°C > threshold={self.config.drift_threshold_c}°C"
                )

        return drifted_blocks

    def flag_directives_for_review(
        self,
        directives: list[ZoningDirective],
        drifted_blocks: dict[str, float],
    ) -> list[ZoningDirective]:
        """Auto-flag directives for blocks with model drift > threshold.

        Annotates each flagged directive with the drift magnitude.

        Args:
            directives: All active zoning directives.
            drifted_blocks: Block IDs with their drift magnitudes.

        Returns:
            List of flagged directives (modified in place with annotations).
        """
        flagged = []
        for directive in directives:
            if directive.block_id in drifted_blocks:
                drift_mag = drifted_blocks[directive.block_id]
                directive.review_trigger = (
                    f"AUTO-FLAGGED: Model drift {drift_mag:.2f}°C detected "
                    f"(threshold: {self.config.drift_threshold_c}°C). "
                    f"Previous trigger: {directive.review_trigger}"
                )
                flagged.append(directive)
                logger.info(
                    f"Directive for block {directive.block_id} flagged: "
                    f"drift={drift_mag:.2f}°C"
                )

        return flagged

    def get_checkpoint_history(self) -> list[CheckpointArtifact]:
        """Return all retained checkpoint artifacts."""
        return list(self._checkpoint_history)

    def load_previous_checkpoint(
        self, index: int = -1
    ) -> PINNModel:
        """Load a previous checkpoint by index in history.

        Args:
            index: Index in checkpoint history (-1 = most recent).

        Returns:
            Loaded PINNModel from the specified checkpoint.
        """
        if not self._checkpoint_history:
            raise ValueError("No checkpoints in history")

        artifact = self._checkpoint_history[index]
        return PINNModel.load_checkpoint(
            Path(artifact.file_path),
            config=self.pinn_config,
            provenance_store=self.provenance,
        )
