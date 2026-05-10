"""Phase 2: Urban Canopy Model Builder.

Computes morphological indices and canyon classifications for every city block
in the Urban Digital Twin. These parameterize how each block exchanges heat
and momentum with the atmosphere.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import numpy as np

from src.shared.types import (
    UCMBlock, UCMArtifact, UDTArtifact, UDTVoxel, ProvenanceRecord,
)
from src.shared.exceptions import UCMValidationError
from src.provenance.store import ProvenanceStore

logger = logging.getLogger(__name__)


class UCMBuilder:
    """Computes morphological indices and canyon classifications for city blocks.

    For each block in the UDT, computes: Sky View Factor (SVF), Plan Area Fraction (λp),
    Frontal Area Index (λf), Mean Aspect Ratio (H/W), and aerodynamic roughness length (z₀).
    Then classifies canyon type and assigns turbulence parameters.
    """

    def __init__(self, provenance_store: ProvenanceStore | None = None) -> None:
        self.provenance = provenance_store or ProvenanceStore()

    def build(
        self,
        udt_artifact: UDTArtifact,
        block_definitions: list[dict[str, Any]] | None = None,
    ) -> UCMArtifact:
        """Compute morphological indices for all blocks and persist.

        Args:
            udt_artifact: The UDT to compute indices for.
            block_definitions: Optional list of block definitions with geometry.
                Each dict should have: block_id, building_heights, block_area_m2,
                footprint_area_m2, mean_building_width_m, street_width_m.

        Returns:
            UCMArtifact with version_id and block statistics.

        Raises:
            UCMValidationError: If input fields are out of physically plausible ranges.
        """
        version_id = str(uuid.uuid4())

        if block_definitions is None:
            block_definitions = self._generate_synthetic_blocks(udt_artifact)

        blocks: list[UCMBlock] = []
        canyon_class_dist: dict[str, int] = {
            "deep_canyon": 0,
            "regular_canyon": 0,
            "shallow_canyon": 0,
            "transitional_zone": 0,
        }

        for block_def in block_definitions:
            block_id = block_def["block_id"]

            # Validate inputs
            self._validate_block_inputs(block_def)

            # Compute morphological indices
            building_heights = block_def.get("building_heights", [10.0])
            block_area = block_def["block_area_m2"]
            footprint_area = block_def.get("footprint_area_m2", block_area * 0.4)
            mean_building_width = block_def.get("mean_building_width_m", 15.0)
            street_width = block_def.get("street_width_m", 12.0)

            mean_height = float(np.mean(building_heights)) if building_heights else 0.0

            # Sky View Factor (SVF)
            svf = self._compute_svf(mean_height, street_width)

            # Plan Area Fraction (λp)
            lambda_p = footprint_area / block_area if block_area > 0 else 0.0

            # Frontal Area Index (λf)
            lambda_f = self._compute_lambda_f(
                building_heights, mean_building_width, block_area
            )

            # Mean Aspect Ratio (H/W)
            hw_ratio = mean_height / street_width if street_width > 0 else 0.0

            # Roughness length (z₀)
            z0 = self._compute_roughness_length(mean_height, lambda_p, lambda_f)

            # Canyon classification
            canyon_class = self._classify_canyon(hw_ratio, block_def)

            # Turbulence params for transitional zones
            turb_params = None
            if canyon_class == "transitional_zone":
                turb_params = self._compute_turbulence_params(
                    building_heights, street_width
                )

            block = UCMBlock(
                block_id=block_id,
                udt_version_id=udt_artifact.version_id,
                svf=float(np.clip(svf, 0, 1)),
                lambda_p=float(np.clip(lambda_p, 0, 1)),
                lambda_f=float(np.clip(lambda_f, 0, 1)),
                hw_ratio=float(max(0, hw_ratio)),
                z0=float(max(0.001, z0)),
                canyon_class=canyon_class,
                turbulence_perturbation_params=turb_params,
                building_heights=building_heights,
                block_area_m2=block_area,
                footprint_area_m2=footprint_area,
            )
            blocks.append(block)
            canyon_class_dist[canyon_class] = canyon_class_dist.get(canyon_class, 0) + 1

        artifact = UCMArtifact(
            version_id=version_id,
            udt_version_id=udt_artifact.version_id,
            block_count=len(blocks),
            creation_timestamp=datetime.utcnow(),
            canyon_class_distribution=canyon_class_dist,
        )

        # Record provenance
        self.provenance.record_artifact(ProvenanceRecord(
            artifact_id=version_id,
            artifact_type="ucm",
            producing_component="ucm_builder",
            input_artifact_ids=[udt_artifact.version_id],
            metadata={
                "block_count": len(blocks),
                "canyon_class_distribution": canyon_class_dist,
            },
        ))

        logger.info(
            f"UCM built: version={version_id}, blocks={len(blocks)}, "
            f"canyon_distribution={canyon_class_dist}"
        )
        return artifact

    def _validate_block_inputs(self, block_def: dict[str, Any]) -> None:
        """Validate that block input fields are within physically plausible ranges."""
        block_id = block_def.get("block_id", "unknown")

        # Validate building heights
        for h in block_def.get("building_heights", []):
            if h < 0 or h > 600:
                raise UCMValidationError(
                    block_id, "building_height", h, "[0, 600] m"
                )

        # Validate block area
        block_area = block_def.get("block_area_m2", 0)
        if block_area <= 0:
            raise UCMValidationError(
                block_id, "block_area_m2", block_area, "> 0 m²"
            )

    def _compute_svf(self, mean_height: float, street_width: float) -> float:
        """Compute Sky View Factor from canyon geometry.

        SVF = cos(atan(2 * H / W)) for symmetric canyon approximation.
        """
        if street_width <= 0:
            return 0.0
        hw = mean_height / street_width
        return float(np.cos(np.arctan(2 * hw)))

    def _compute_lambda_f(
        self,
        building_heights: list[float],
        mean_width: float,
        block_area: float,
    ) -> float:
        """Compute Frontal Area Index (building silhouette per unit ground area)."""
        if block_area <= 0 or not building_heights:
            return 0.0
        total_frontal = sum(h * mean_width for h in building_heights)
        return total_frontal / block_area

    def _compute_roughness_length(
        self, mean_height: float, lambda_p: float, lambda_f: float
    ) -> float:
        """Compute aerodynamic roughness length using Macdonald's formula.

        z₀ = mean_height × (1 - d/h) × exp(-[0.5 × β × Cd/κ² × (1 - d/h) × λf]^(-0.5))
        Simplified: z₀ ≈ 0.1 × mean_height × λf for moderate density.
        """
        if mean_height <= 0 or lambda_f <= 0:
            return 0.01  # minimal roughness
        # Simplified Macdonald model
        d_over_h = min(lambda_p * 0.7, 0.9)  # displacement height ratio
        z0 = mean_height * (1 - d_over_h) * 0.1 * lambda_f
        return max(0.001, z0)

    def _classify_canyon(
        self, hw_ratio: float, block_def: dict[str, Any]
    ) -> str:
        """Classify canyon type based on H/W aspect ratio.

        - deep_canyon:        H/W > 2.5
        - regular_canyon:     1 ≤ H/W ≤ 2.5
        - shallow_canyon:     H/W < 1
        - transitional_zone:  variable H/W between adjacent blocks
        """
        # Check if this is a transitional zone (variable H/W)
        if block_def.get("is_transitional", False):
            return "transitional_zone"

        adjacent_hw = block_def.get("adjacent_hw_ratios", [])
        if adjacent_hw:
            hw_std = float(np.std([hw_ratio] + adjacent_hw))
            if hw_std > 0.5:  # High variability indicates transitional zone
                return "transitional_zone"

        if hw_ratio > 2.5:
            return "deep_canyon"
        elif hw_ratio >= 1.0:
            return "regular_canyon"
        else:
            return "shallow_canyon"

    def _compute_turbulence_params(
        self, building_heights: list[float], street_width: float
    ) -> dict[str, float]:
        """Compute stochastic turbulence perturbation parameters for transitional zones."""
        height_std = float(np.std(building_heights)) if building_heights else 0.0
        mean_height = float(np.mean(building_heights)) if building_heights else 0.0

        return {
            "height_variability": height_std,
            "mean_height_m": mean_height,
            "street_width_m": street_width,
            "turbulence_intensity": min(1.0, height_std / max(mean_height, 1.0)),
            "perturbation_scale": height_std * 0.1,
        }

    def _generate_synthetic_blocks(
        self, udt_artifact: UDTArtifact
    ) -> list[dict[str, Any]]:
        """Generate synthetic block definitions for development/testing."""
        rng = np.random.default_rng(42)
        blocks = []
        n_blocks = 20

        for i in range(n_blocks):
            n_buildings = rng.integers(3, 15)
            heights = rng.uniform(5, 60, n_buildings).tolist()
            block_area = rng.uniform(2000, 10000)

            blocks.append({
                "block_id": f"BLK-{i + 1:02d}",
                "building_heights": heights,
                "block_area_m2": block_area,
                "footprint_area_m2": block_area * rng.uniform(0.2, 0.6),
                "mean_building_width_m": rng.uniform(8, 25),
                "street_width_m": rng.uniform(6, 20),
                "adjacent_hw_ratios": rng.uniform(0.5, 3.0, rng.integers(1, 4)).tolist(),
            })
        return blocks
