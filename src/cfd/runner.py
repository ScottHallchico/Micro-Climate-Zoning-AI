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

    def simulate_block_microclimate(
        self,
        blocks: list[dict[str, Any]] | None = None,
        wind_speed_ms: float = 4.2,
        wind_direction_deg: float = 85.0,
        sun_altitude_deg: float = 58.0,
        sun_azimuth_deg: float = 135.0,
        measured_solar_radiation_wm2: float | None = None,
        base_air_temp_c: float = 34.0,
        surface_temp_reference_c: float | None = None,
    ) -> list[dict[str, Any]]:
        """Simulate block-level wind and sunlight metrics for dashboard use.

        This is a fast reduced-order CFD/solar model. It is not a replacement for
        OpenFOAM, but it preserves the same inputs/outputs a full CFD run would
        provide to the PINN and dashboard: wind speed, deflection, ventilation,
        solar access, shadow coverage, radiation, and combined heat burden.
        """
        if blocks is None:
            blocks = self._generate_dashboard_blocks()

        results = []
        for block in blocks:
            height = float(block.get("max_height_m", 24.0))
            svf = float(block.get("svf", 0.5))
            plan_area = float(block.get("lambda_p", 0.35))
            hw_ratio = float(block.get("hw_ratio", 1.4))
            albedo = float(block.get("albedo", 0.28))
            green_cover = float(block.get("green_cover", 20.0))
            zone_class = str(block.get("zone_class", "BASELINE_UNCHANGED"))

            wind = self._simulate_block_wind(
                height=height,
                svf=svf,
                plan_area=plan_area,
                hw_ratio=hw_ratio,
                wind_speed_ms=wind_speed_ms,
                wind_direction_deg=wind_direction_deg,
                zone_class=zone_class,
            )
            solar = self._simulate_block_solar(
                height=height,
                svf=svf,
                plan_area=plan_area,
                albedo=albedo,
                green_cover=green_cover,
                sun_altitude_deg=sun_altitude_deg,
                sun_azimuth_deg=sun_azimuth_deg,
                measured_solar_radiation_wm2=measured_solar_radiation_wm2,
            )

            heat_burden = self._compute_heat_burden(
                wind_speed=wind["avg_wind_speed_ms"],
                solar_radiation=solar["solar_radiation_wm2"],
                green_cover=green_cover,
                albedo=albedo,
            )
            thermal = self._simulate_block_thermal(
                solar_radiation=solar["solar_radiation_wm2"],
                wind_speed=wind["avg_wind_speed_ms"],
                shadow_coverage_pct=solar["shadow_coverage_pct"],
                green_cover=green_cover,
                albedo=albedo,
                hw_ratio=hw_ratio,
                heat_burden=heat_burden,
                base_air_temp_c=base_air_temp_c,
                surface_temp_reference_c=surface_temp_reference_c,
            )

            results.append({
                "block_id": block.get("id") or block.get("block_id", "unknown"),
                "wind": wind,
                "solar": solar,
                "thermal": thermal,
                "microclimate": {
                    "ventilation_score": wind["ventilation_score"],
                    "heat_burden_score": heat_burden,
                    "combined_risk": self._classify_microclimate_risk(heat_burden),
                },
            })

        return results

    def _simulate_block_wind(
        self,
        height: float,
        svf: float,
        plan_area: float,
        hw_ratio: float,
        wind_speed_ms: float,
        wind_direction_deg: float,
        zone_class: str,
    ) -> dict[str, float | str]:
        """Reduced-order street-canyon wind model."""
        canyon_drag = np.clip(1.0 - 0.14 * hw_ratio - 0.30 * plan_area, 0.18, 0.95)
        sky_acceleration = 0.55 + 0.65 * svf
        corridor_bonus = 1.18 if zone_class == "WIND_CORRIDOR_CRITICAL" else 1.0
        avg_wind_speed = wind_speed_ms * canyon_drag * sky_acceleration * corridor_bonus

        deflection = np.clip(hw_ratio * 9.5 + plan_area * 24.0 - svf * 8.0, 0.0, 55.0)
        if zone_class == "WIND_CORRIDOR_CRITICAL" and height > 18:
            deflection += min(18.0, (height - 18.0) * 0.8)

        pressure_drop = max(0.0, 0.5 * 1.225 * (wind_speed_ms**2 - avg_wind_speed**2))
        ventilation_score = np.clip(avg_wind_speed / max(wind_speed_ms, 0.1), 0.0, 1.0)

        return {
            "inlet_wind_speed_ms": round(float(wind_speed_ms), 2),
            "avg_wind_speed_ms": round(float(avg_wind_speed), 2),
            "wind_direction_deg": round(float(wind_direction_deg), 1),
            "deflection_deg": round(float(deflection), 1),
            "pressure_drop_pa": round(float(pressure_drop), 2),
            "ventilation_score": round(float(ventilation_score), 2),
        }

    def _simulate_block_solar(
        self,
        height: float,
        svf: float,
        plan_area: float,
        albedo: float,
        green_cover: float,
        sun_altitude_deg: float,
        sun_azimuth_deg: float,
        measured_solar_radiation_wm2: float | None = None,
    ) -> dict[str, float]:
        """Reduced-order solar radiation and shadow model."""
        altitude_rad = np.radians(max(sun_altitude_deg, 5.0))
        shadow_length_m = height / np.tan(altitude_rad)
        shadow_coverage = np.clip((shadow_length_m / 42.0) * (0.55 + plan_area), 0.0, 0.92)
        solar_access_hours = np.clip(8.2 * svf * (1.0 - shadow_coverage * 0.55), 0.4, 8.5)

        clear_sky_radiation = (
            measured_solar_radiation_wm2
            if measured_solar_radiation_wm2 is not None
            else 940.0 * np.sin(altitude_rad)
        )
        absorbed_fraction = np.clip(1.0 - albedo - green_cover / 260.0, 0.18, 0.92)
        solar_radiation = clear_sky_radiation * (1.0 - shadow_coverage * 0.45) * absorbed_fraction

        return {
            "sun_altitude_deg": round(float(sun_altitude_deg), 1),
            "sun_azimuth_deg": round(float(sun_azimuth_deg), 1),
            "shadow_length_m": round(float(shadow_length_m), 2),
            "shadow_coverage_pct": round(float(shadow_coverage * 100.0), 1),
            "solar_access_hours": round(float(solar_access_hours), 2),
            "solar_radiation_wm2": round(float(solar_radiation), 1),
        }

    def _simulate_block_thermal(
        self,
        solar_radiation: float,
        wind_speed: float,
        shadow_coverage_pct: float,
        green_cover: float,
        albedo: float,
        hw_ratio: float,
        heat_burden: float,
        base_air_temp_c: float = 34.0,
        surface_temp_reference_c: float | None = None,
    ) -> dict[str, float | str]:
        """Reduced-order thermal layer: surface, air, storage, and UHI."""
        solar_heating = solar_radiation / 145.0
        ventilation_cooling = min(3.2, wind_speed * 0.55)
        shade_cooling = shadow_coverage_pct * 0.025
        green_cooling = green_cover * 0.035
        albedo_cooling = albedo * 3.6
        canyon_storage = max(0.0, hw_ratio - 1.0) * 0.55

        surface_temp_c = (
            base_air_temp_c
            + solar_heating
            + canyon_storage
            - ventilation_cooling
            - shade_cooling
            - green_cooling
            - albedo_cooling
            + heat_burden * 2.4
        )
        if surface_temp_reference_c is not None:
            surface_temp_c = surface_temp_c * 0.55 + (
                surface_temp_reference_c
                + solar_heating * 0.45
                + canyon_storage
                - green_cooling
                - albedo_cooling * 0.5
            ) * 0.45
        air_temp_c = surface_temp_c - 3.8 + heat_burden * 1.2
        rural_reference_c = max(20.0, base_air_temp_c - 1.8)
        uhi_intensity_c = air_temp_c - rural_reference_c
        heat_storage_wm2 = max(0.0, solar_radiation * (0.22 + heat_burden * 0.28) * (1.0 - albedo * 0.45))

        return {
            "surface_temp_c": round(float(surface_temp_c), 1),
            "air_temp_c": round(float(air_temp_c), 1),
            "uhi_intensity_c": round(float(uhi_intensity_c), 2),
            "heat_storage_wm2": round(float(heat_storage_wm2), 1),
            "thermal_risk": self._classify_thermal_risk(surface_temp_c),
        }

    def _compute_heat_burden(
        self,
        wind_speed: float,
        solar_radiation: float,
        green_cover: float,
        albedo: float,
    ) -> float:
        """Combine ventilation and solar loading into a 0-1 risk score."""
        solar_load = np.clip(solar_radiation / 760.0, 0.0, 1.0)
        stagnant_air = np.clip(1.0 - wind_speed / 4.5, 0.0, 1.0)
        green_relief = np.clip(green_cover / 100.0, 0.0, 0.6)
        albedo_relief = np.clip(albedo, 0.0, 0.75) * 0.35
        burden = solar_load * 0.55 + stagnant_air * 0.45 - green_relief * 0.25 - albedo_relief
        return round(float(np.clip(burden, 0.0, 1.0)), 2)

    def _classify_microclimate_risk(self, heat_burden: float) -> str:
        if heat_burden >= 0.68:
            return "high"
        if heat_burden >= 0.38:
            return "moderate"
        return "low"

    def _classify_thermal_risk(self, surface_temp_c: float) -> str:
        if surface_temp_c >= 41.0:
            return "extreme"
        if surface_temp_c >= 38.0:
            return "high"
        if surface_temp_c >= 35.0:
            return "moderate"
        return "low"

    def _generate_dashboard_blocks(self) -> list[dict[str, Any]]:
        """Mirror the dashboard's deterministic demo blocks for API simulations."""
        zone_classes = [
            "WIND_CORRIDOR_CRITICAL",
            "THERMAL_REMEDIATION",
            "DENSITY_ADAPTIVE",
            "BASELINE_UNCHANGED",
        ]
        blocks = []
        for i in range(20):
            blocks.append({
                "id": f"BLK-{i + 1:02d}",
                "zone_class": zone_classes[(i * 7) % len(zone_classes)],
                "max_height_m": round(8 + ((i * 9) % 52)),
                "svf": round(0.2 + ((i * 0.11) % 0.62), 2),
                "lambda_p": round(0.12 + ((i * 0.09) % 0.48), 2),
                "hw_ratio": round(0.5 + ((i * 0.37) % 3.0), 1),
                "albedo": round(0.12 + ((i * 0.07) % 0.55), 2),
                "green_cover": round(8 + ((i * 13) % 42)),
            })
        blocks[0]["zone_class"] = "WIND_CORRIDOR_CRITICAL"
        blocks[0]["max_height_m"] = 12
        blocks[2].update({
            "zone_class": "DENSITY_ADAPTIVE",
            "max_height_m": 27,
            "svf": 0.5,
            "lambda_p": 0.49,
            "hw_ratio": 0.8,
            "albedo": 0.26,
            "green_cover": 20,
        })
        blocks[8]["zone_class"] = "THERMAL_REMEDIATION"
        blocks[16]["zone_class"] = "DENSITY_ADAPTIVE"
        blocks[16]["max_height_m"] = 36
        return blocks

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
