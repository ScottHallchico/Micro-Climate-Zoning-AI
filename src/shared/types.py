"""Shared data models and type definitions for the Micro-Climate Zoning AI pipeline.

All dataclasses here mirror the design document's data model specifications exactly.
These are the canonical types consumed and produced by every phase of the pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CanyonClass(str, Enum):
    """Canyon type classification based on H/W aspect ratio."""
    DEEP_CANYON = "deep_canyon"
    REGULAR_CANYON = "regular_canyon"
    SHALLOW_CANYON = "shallow_canyon"
    TRANSITIONAL_ZONE = "transitional_zone"


class ZoneClass(str, Enum):
    """Zoning classification assigned by the optimizer."""
    WIND_CORRIDOR_CRITICAL = "WIND_CORRIDOR_CRITICAL"
    THERMAL_REMEDIATION = "THERMAL_REMEDIATION"
    DENSITY_ADAPTIVE = "DENSITY_ADAPTIVE"
    BASELINE_UNCHANGED = "BASELINE_UNCHANGED"


class ComplianceStatus(str, Enum):
    """Permit compliance evaluation outcome."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    CONDITIONAL = "conditional"


class GeometryType(str, Enum):
    """Voxel geometry classification."""
    BUILDING = "building"
    GROUND = "ground"
    VEGETATION = "vegetation"
    AIR = "air"


class ArtifactType(str, Enum):
    """Types of versioned artifacts produced by the pipeline."""
    UDT = "udt"
    CFD_DATASET = "cfd_dataset"
    PINN_CHECKPOINT = "pinn_checkpoint"
    PARETO_FRONT = "pareto_front"
    ZONING_DIRECTIVE = "zoning_directive"
    UCM = "ucm"


class CFDSampleStatus(str, Enum):
    """Validation status of a CFD simulation run."""
    VALID = "valid"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Phase 1: Data Ingestion
# ---------------------------------------------------------------------------

@dataclass
class UDTVoxel:
    """A single cell in the Urban Digital Twin voxel grid (2m × 2m × 2m)."""
    x: float              # easting (m)
    y: float              # northing (m)
    z: float              # elevation (m)
    geometry_type: str    # "building" | "ground" | "vegetation" | "air"
    albedo: float         # α ∈ [0, 1]
    emissivity: float     # ε ∈ [0, 1]
    heat_capacity: float  # ρCp (J/m³/K)
    boundary_temp_k: float  # boundary thermal condition (K)
    material_class: str


@dataclass
class UDTArtifact:
    """Versioned reference to a complete Urban Digital Twin stored in PostGIS."""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spatial_extent: dict[str, float] = field(default_factory=dict)  # {min_x, max_x, min_y, max_y, min_z, max_z}
    postgis_table: str = "udt_voxels"
    voxel_resolution_m: float = 2.0
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    source_datasets: list[str] = field(default_factory=list)
    preprocessing_params: dict[str, Any] = field(default_factory=dict)
    voxel_count: int = 0


# ---------------------------------------------------------------------------
# Phase 2: Urban Canopy Model
# ---------------------------------------------------------------------------

@dataclass
class UCMBlock:
    """Morphological indices and canyon classification for a single city block."""
    block_id: str
    udt_version_id: str
    svf: float              # Sky View Factor ∈ [0, 1]
    lambda_p: float         # Plan Area Fraction ∈ [0, 1]
    lambda_f: float         # Frontal Area Index ∈ [0, 1]
    hw_ratio: float         # Mean Aspect Ratio H/W ≥ 0
    z0: float               # Roughness length (m) > 0
    canyon_class: str        # "deep_canyon" | "regular_canyon" | "shallow_canyon" | "transitional_zone"
    turbulence_perturbation_params: dict | None = None  # set for transitional_zone only
    building_heights: list[float] = field(default_factory=list)
    block_area_m2: float = 0.0
    footprint_area_m2: float = 0.0


@dataclass
class UCMArtifact:
    """Versioned reference to a complete UCM stored in PostGIS."""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    udt_version_id: str = ""
    postgis_table: str = "ucm_blocks"
    block_count: int = 0
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    canyon_class_distribution: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 2: CFD Training Data
# ---------------------------------------------------------------------------

@dataclass
class CFDSample:
    """A single CFD simulation result for PINN training."""
    sample_id: str
    dataset_version_id: str
    morphology_params: dict           # UCM indices used as input
    wind_field: np.ndarray | None = None    # shape (Nx, Ny, Nz, 4) — (u, v, w, P)
    energy_field: np.ndarray | None = None  # shape (Nx, Ny, Nz, 4) — (Qh, Qe, ΔQs, Q*)
    mass_conservation_residual: float = 0.0
    status: str = "valid"            # "valid" | "failed"


@dataclass
class CFDDatasetArtifact:
    """Versioned reference to the complete CFD training dataset."""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ucm_version_id: str = ""
    sample_count: int = 0
    valid_sample_count: int = 0
    failed_sample_count: int = 0
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    samples: list[CFDSample] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 3: PINN Model
# ---------------------------------------------------------------------------

@dataclass
class CheckpointManifest:
    """Metadata for a serialized PINN checkpoint."""
    version_id: str
    training_dataset_version: str
    fine_tuning_city_id: str | None = None
    training_timestamp: datetime = field(default_factory=datetime.utcnow)
    architecture_config: dict = field(default_factory=dict)  # layer counts, widths, activation type
    physics_coefficients: dict = field(default_factory=dict)  # {"lambda_1": float, "lambda_2": float, "lambda_3": float}
    validation_mae: float = 0.0


@dataclass
class CheckpointArtifact:
    """Versioned reference to a serialized PINN checkpoint file."""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    manifest: CheckpointManifest | None = None
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PINNPrediction:
    """Output of a PINN forward pass for a single block configuration."""
    wind_field: np.ndarray | None = None    # (u, v, w, P) at every voxel
    energy_field: np.ndarray | None = None  # (Qh, Qe, ΔQs, Q*) at every voxel
    ood_warning: bool = False               # True if any input feature is > 3σ from training mean
    ood_features: list[str] = field(default_factory=list)  # Names of out-of-distribution features
    block_temperature: float = 0.0          # Predicted block surface temperature


@dataclass
class WeatherMeasurement:
    """A single weather station measurement for PINN fine-tuning."""
    station_id: str
    timestamp: datetime
    temperature_k: float      # Kelvin
    wind_speed_ms: float      # m/s
    wind_direction_deg: float # degrees from north
    humidity_pct: float       # 0-100
    solar_irradiance_wm2: float  # W/m²
    location_x: float        # easting (m)
    location_y: float        # northing (m)


# ---------------------------------------------------------------------------
# Phase 4: Optimization
# ---------------------------------------------------------------------------

@dataclass
class CorridorGraph:
    """Graph representation of wind corridors for protection analysis."""
    nodes: list[str] = field(default_factory=list)         # Street segment IDs
    edges: list[tuple[str, str, float]] = field(default_factory=list)  # (node1, node2, wind_weight)
    bottleneck_blocks: list[str] = field(default_factory=list)  # Critical blocks with height caps
    height_caps: dict[str, float] = field(default_factory=dict)  # block_id -> max_height_m
    corridor_vectors: dict[str, dict] = field(default_factory=dict)  # block_id -> {bearing_deg, avg_wind_speed_ms}


@dataclass
class OptimizationConstraints:
    """Hard and soft constraints for the MOBO optimizer."""
    far_max: dict[str, float] = field(default_factory=dict)      # block_id -> max FAR
    delta_h_max: dict[str, float] = field(default_factory=dict)  # block_id -> max height change (m)
    v_min_corridors: dict[str, float] = field(default_factory=dict)  # corridor_id -> min wind speed (m/s)
    min_green_cover_fraction: float = 0.15  # 15% minimum
    budget_total: float = 0.0
    t_rural: float = 300.0  # Rural reference temperature (K)
    pop_density_weights: dict[str, float] = field(default_factory=dict)  # block_id -> population density weight


@dataclass
class ParetoConfiguration:
    """A single Pareto-optimal urban configuration."""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    decision_variables: dict[str, dict] = field(default_factory=dict)  # block_id -> {height, setback, albedo, ...}
    uhi_intensity: float = 0.0       # Primary objective value
    pet_index: float = 0.0           # Pedestrian thermal comfort
    solar_access_hours: float = 0.0  # Minimum solar access across blocks
    retrofit_cost: float = 0.0       # Total retrofit cost
    block_temperatures: dict[str, float] = field(default_factory=dict)  # block_id -> predicted T
    dominates: list[str] = field(default_factory=list)  # IDs of dominated configs


@dataclass
class ParetoFrontArtifact:
    """Versioned reference to a complete Pareto front of urban configurations."""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ucm_version_id: str = ""
    pinn_checkpoint_version: str = ""
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    configurations: list[ParetoConfiguration] = field(default_factory=list)
    constraint_params: dict = field(default_factory=dict)
    config_count: int = 0


# ---------------------------------------------------------------------------
# Phase 5: Zoning Generation
# ---------------------------------------------------------------------------

@dataclass
class MandatoryIntervention:
    """A mandatory physical intervention required by a zoning directive."""
    type: str                       # "green_roof" | "pavement_albedo" | "street_tree_canopy" | etc.
    cooling_effect_estimate_c: float  # Estimated cooling in °C
    min_coverage_pct: float | None = None
    min_albedo: float | None = None
    species: str | None = None
    placement: str | None = None
    current_value: float | None = None
    retrofit_priority: str | None = None


@dataclass
class DirectiveProvenance:
    """Provenance trail for a single zoning directive."""
    udt_version_id: str
    pinn_checkpoint_version: str
    cfd_sample_ids: list[str]
    mobo_run_id: str
    ood_warning: bool = False  # True if directive was based on OOD prediction
    ood_features: list[str] = field(default_factory=list)  # Names of OOD features


@dataclass
class ZoningDirective:
    """A complete zoning directive for a single city block."""
    block_id: str
    zone_class: str          # WIND_CORRIDOR_CRITICAL | THERMAL_REMEDIATION | DENSITY_ADAPTIVE | BASELINE_UNCHANGED
    max_height_m: float | None
    height_justification: str
    permitted_uses: list[str]
    mandatory_interventions: list[MandatoryIntervention]
    compliance_deadline_months: int
    review_trigger: str
    provenance: DirectiveProvenance | None = None
    # Wind corridor specific fields (for WIND_CORRIDOR_CRITICAL)
    corridor_vector_bearing_deg: float | None = None
    corridor_avg_wind_speed_ms: float | None = None
    downstream_cooling_loss_c: float | None = None
    # Metadata
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    optimization_timestamp: datetime | None = None


# ---------------------------------------------------------------------------
# Phase 5: Governance API
# ---------------------------------------------------------------------------

@dataclass
class PermitApplication:
    """A building permit application payload for compliance checking."""
    block_id: str
    proposed_height_m: float
    proposed_far: float
    proposed_roof_albedo: float
    proposed_green_cover_pct: float  # 0–100


@dataclass
class ComplianceResponse:
    """Result of a permit compliance check."""
    compliance_status: str   # "compliant" | "non_compliant" | "conditional"
    violated_fields: list[str]  # empty if compliant
    active_directive_version_id: str
    checked_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceRecord:
    """A single record in the central provenance store."""
    artifact_id: str
    artifact_type: str        # "udt" | "cfd_dataset" | "pinn_checkpoint" | "pareto_front" | "zoning_directive" | "ucm"
    producing_component: str  # "ingestion_pipeline" | "ucm_builder" | "cfd_runner" | "pinn_model" | "optimizer" | "zoning_generator"
    input_artifact_ids: list[str]
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Community Heat Index
# ---------------------------------------------------------------------------

@dataclass
class WardHeatIndex:
    """Per-ward aggregate metrics for equity monitoring."""
    ward_id: str
    total_cooling_benefit_c_m2: float   # Σ(ΔT × block_area)
    total_retrofit_cost: float
    displacement_risk_score: float      # 0–1
    equity_review_flagged: bool = False


@dataclass
class WardMap:
    """Mapping of blocks to wards for community heat index computation."""
    ward_blocks: dict[str, list[str]] = field(default_factory=dict)  # ward_id -> [block_id, ...]
    ward_names: dict[str, str] = field(default_factory=dict)         # ward_id -> display name
    displacement_risk_threshold: float = 0.5
