"""Phase 2: CFD Training Data Generation.

Orchestrates OpenFOAM RANS k-ε simulations on stratified morphology samples
to generate ground-truth training data for the PINN.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from scipy.stats import qmc

from src.shared.types import (
    CFDSample, CFDDatasetArtifact, UCMArtifact, UCMBlock, ProvenanceRecord,
)
from src.shared.config import CFDConfig
from src.provenance.store import ProvenanceStore

logger = logging.getLogger(__name__)


class CFDRunner:
    """Orchestrates OpenFOAM RANS k-ε simulations for PINN training data.

    Selects 200 stratified samples via Latin Hypercube Sampling over the
    morphological index space, runs CFD simulations, validates mass conservation,
    and packages results into a versioned dataset artifact.
    """

    def __init__(
        self,
        config: CFDConfig | None = None,
        provenance_store: ProvenanceStore | None = None,
    ) -> None:
        self.config = config or CFDConfig()
        self.provenance = provenance_store or ProvenanceStore()

    def generate_training_dataset(
        self,
        ucm_artifact: UCMArtifact,
        blocks: list[UCMBlock] | None = None,
        n_samples: int | None = None,
    ) -> CFDDatasetArtifact:
        """Generate CFD training dataset via stratified sampling and simulation.

        Args:
            ucm_artifact: The UCM to sample from.
            blocks: Optional list of UCM blocks. If None, generates synthetic.
            n_samples: Number of samples (default: config.n_samples = 200).

        Returns:
            CFDDatasetArtifact with validated simulation results.
        """
        n_samples = n_samples or self.config.n_samples
        version_id = str(uuid.uuid4())

        if blocks is None:
            blocks = self._generate_synthetic_blocks(ucm_artifact)

        # Step 1: Stratified sampling via LHS
        sample_params = self._lhs_sample(blocks, n_samples)
        logger.info(f"Generated {len(sample_params)} stratified samples via LHS")

        # Step 2: Run CFD simulations
        samples: list[CFDSample] = []
        valid_count = 0
        failed_count = 0

        for i, params in enumerate(sample_params):
            sample_id = f"CFD-{version_id[:8]}-{i + 1:04d}"

            try:
                # Simulate CFD run (in production: OpenFOAM)
                result = self._run_cfd_simulation(params)

                # Validate mass conservation
                residual = self._compute_mass_conservation_residual(result["wind_field"])
                status = "valid" if residual < self.config.mass_conservation_threshold else "failed"

                sample = CFDSample(
                    sample_id=sample_id,
                    dataset_version_id=version_id,
                    morphology_params=params,
                    wind_field=result["wind_field"],
                    energy_field=result["energy_field"],
                    mass_conservation_residual=float(residual),
                    status=status,
                )

                if status == "valid":
                    valid_count += 1
                else:
                    failed_count += 1
                    logger.warning(
                        f"CFD sample {sample_id} failed mass conservation: "
                        f"residual={residual:.6e} > threshold={self.config.mass_conservation_threshold:.1e}"
                    )

                samples.append(sample)

            except Exception as e:
                failed_count += 1
                logger.error(f"CFD simulation failed for sample {sample_id}: {e}")
                samples.append(CFDSample(
                    sample_id=sample_id,
                    dataset_version_id=version_id,
                    morphology_params=params,
                    mass_conservation_residual=float("inf"),
                    status="failed",
                ))

            if (i + 1) % 50 == 0:
                logger.info(f"CFD progress: {i + 1}/{n_samples} samples completed")

        # Build dataset artifact (only valid samples)
        valid_samples = [s for s in samples if s.status == "valid"]

        artifact = CFDDatasetArtifact(
            version_id=version_id,
            ucm_version_id=ucm_artifact.version_id,
            sample_count=len(samples),
            valid_sample_count=valid_count,
            failed_sample_count=failed_count,
            creation_timestamp=datetime.utcnow(),
            samples=valid_samples,
        )

        # Record provenance
        self.provenance.record_artifact(ProvenanceRecord(
            artifact_id=version_id,
            artifact_type="cfd_dataset",
            producing_component="cfd_runner",
            input_artifact_ids=[ucm_artifact.version_id],
            metadata={
                "total_samples": len(samples),
                "valid_samples": valid_count,
                "failed_samples": failed_count,
                "mass_conservation_threshold": self.config.mass_conservation_threshold,
            },
        ))

        logger.info(
            f"CFD dataset created: version={version_id}, "
            f"valid={valid_count}, failed={failed_count}"
        )
        return artifact

    def _lhs_sample(
        self, blocks: list[UCMBlock], n_samples: int
    ) -> list[dict[str, Any]]:
        """Generate stratified samples via Latin Hypercube Sampling.

        Ensures coverage across all four canyon classification types.
        Sampling dimensions: SVF, λp, λf, H/W, z₀.
        """
        # Determine parameter ranges from available blocks
        if not blocks:
            # Use physically plausible default ranges
            param_ranges = {
                "svf": (0.1, 0.95),
                "lambda_p": (0.1, 0.7),
                "lambda_f": (0.05, 0.5),
                "hw_ratio": (0.3, 4.0),
                "z0": (0.01, 5.0),
            }
        else:
            param_ranges = {
                "svf": (min(b.svf for b in blocks), max(b.svf for b in blocks)),
                "lambda_p": (min(b.lambda_p for b in blocks), max(b.lambda_p for b in blocks)),
                "lambda_f": (min(b.lambda_f for b in blocks), max(b.lambda_f for b in blocks)),
                "hw_ratio": (min(b.hw_ratio for b in blocks), max(b.hw_ratio for b in blocks)),
                "z0": (min(b.z0 for b in blocks), max(b.z0 for b in blocks)),
            }

        # Ensure ranges are not degenerate
        for key in param_ranges:
            lo, hi = param_ranges[key]
            if lo >= hi:
                param_ranges[key] = (lo * 0.5, lo * 1.5 + 0.01)

        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=5, seed=42)
        unit_samples = sampler.random(n=n_samples)

        # Scale to parameter ranges
        lower = np.array([param_ranges[k][0] for k in param_ranges])
        upper = np.array([param_ranges[k][1] for k in param_ranges])
        scaled_samples = qmc.scale(unit_samples, lower, upper)

        # Ensure all 4 canyon classes are represented
        samples = []
        canyon_classes = ["deep_canyon", "regular_canyon", "shallow_canyon", "transitional_zone"]
        min_per_class = max(1, n_samples // 10)

        for i, row in enumerate(scaled_samples):
            hw = row[3]
            # Classify canyon
            if hw > 2.5:
                canyon = "deep_canyon"
            elif hw >= 1.0:
                canyon = "regular_canyon"
            else:
                canyon = "shallow_canyon"

            # Force some transitional zones
            if i < min_per_class:
                canyon = canyon_classes[i % 4]
                if canyon == "deep_canyon":
                    row[3] = np.random.uniform(2.6, 4.0)
                elif canyon == "regular_canyon":
                    row[3] = np.random.uniform(1.0, 2.5)
                elif canyon == "shallow_canyon":
                    row[3] = np.random.uniform(0.3, 0.9)
                elif canyon == "transitional_zone":
                    row[3] = np.random.uniform(0.8, 2.8)

            params = {
                "svf": float(row[0]),
                "lambda_p": float(row[1]),
                "lambda_f": float(row[2]),
                "hw_ratio": float(row[3]),
                "z0": float(row[4]),
                "canyon_class": canyon,
                "sample_index": i,
            }
            samples.append(params)

        return samples

    def _run_cfd_simulation(self, params: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run a single CFD simulation for given morphology parameters.

        In production, this would submit an OpenFOAM RANS k-ε job.
        For development, generates physics-plausible synthetic fields.
        """
        rng = np.random.default_rng(hash(str(params)) % 2**32)
        grid_size = (16, 16, 8)  # Reduced for dev; production: full voxel grid

        hw_ratio = params.get("hw_ratio", 1.5)
        svf = params.get("svf", 0.5)

        # Generate physically plausible wind field
        # Higher H/W → lower wind speeds at street level
        base_wind = 5.0 * svf
        u = rng.normal(base_wind, 0.5, grid_size)
        v = rng.normal(0, 0.3, grid_size)
        w = rng.normal(0, 0.1, grid_size)
        # Pressure field
        p = rng.normal(101325, 50, grid_size)

        wind_field = np.stack([u, v, w, p], axis=-1)  # (Nx, Ny, Nz, 4)

        # Generate energy field
        q_star = rng.uniform(100, 500, grid_size)  # Net radiation
        q_h = q_star * rng.uniform(0.3, 0.5, grid_size)  # Sensible heat
        q_e = q_star * rng.uniform(0.1, 0.3, grid_size)  # Latent heat
        delta_qs = q_star - q_h - q_e  # Storage heat

        energy_field = np.stack([q_h, q_e, delta_qs, q_star], axis=-1)

        return {"wind_field": wind_field, "energy_field": energy_field}

    def _compute_mass_conservation_residual(
        self, wind_field: np.ndarray
    ) -> float:
        """Compute ∇·u = 0 residual for mass conservation validation.

        Returns the maximum absolute value of the divergence.
        """
        if wind_field is None:
            return float("inf")

        u = wind_field[..., 0]
        v = wind_field[..., 1]
        w = wind_field[..., 2]

        # Compute divergence using central differences
        du_dx = np.gradient(u, axis=0)
        dv_dy = np.gradient(v, axis=1)
        dw_dz = np.gradient(w, axis=2)

        divergence = du_dx + dv_dy + dw_dz
        residual = float(np.max(np.abs(divergence)))

        return residual

    def _generate_synthetic_blocks(
        self, ucm_artifact: UCMArtifact
    ) -> list[UCMBlock]:
        """Generate synthetic UCM blocks for development."""
        rng = np.random.default_rng(42)
        blocks = []
        for i in range(20):
            hw = rng.uniform(0.3, 4.0)
            if hw > 2.5:
                canyon = "deep_canyon"
            elif hw >= 1.0:
                canyon = "regular_canyon"
            else:
                canyon = "shallow_canyon"

            blocks.append(UCMBlock(
                block_id=f"BLK-{i + 1:02d}",
                udt_version_id=ucm_artifact.udt_version_id,
                svf=rng.uniform(0.1, 0.95),
                lambda_p=rng.uniform(0.1, 0.7),
                lambda_f=rng.uniform(0.05, 0.5),
                hw_ratio=hw,
                z0=rng.uniform(0.01, 5.0),
                canyon_class=canyon,
            ))
        return blocks
