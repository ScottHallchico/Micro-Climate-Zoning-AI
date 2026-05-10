"""Configuration management for the Micro-Climate Zoning AI pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class DatabaseConfig:
    """PostGIS database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "microclimate"
    user: str = "microclimate"
    password: str = "microclimate"
    schema: str = "public"

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class PINNConfig:
    """Configuration for the PINN model architecture and training."""
    encoder_layers: int = 8
    encoder_width: int = 512
    activation: str = "siren"  # SIREN sin activations
    wind_head_outputs: int = 4  # u, v, w, P
    energy_head_outputs: int = 4  # Qh, Qe, ΔQs, Q*
    lambda_1: float = 1.0  # Navier-Stokes loss weight
    lambda_2: float = 1.0  # Energy balance loss weight
    lambda_3: float = 1.0  # Boundary condition loss weight
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 1000
    ood_sigma_threshold: float = 3.0  # Standard deviations for OOD detection
    fine_tune_layers: int = 2  # Number of layers to fine-tune in transfer learning
    max_inference_time_s: float = 0.3  # Performance target


@dataclass
class CFDConfig:
    """Configuration for CFD simulation orchestration."""
    n_samples: int = 200
    max_wall_clock_hours: int = 72
    mass_conservation_threshold: float = 1e-4
    openfoam_solver: str = "simpleFoam"
    turbulence_model: str = "kEpsilon"


@dataclass
class OptimizerConfig:
    """Configuration for the MOBO optimization engine."""
    min_pareto_configs: int = 50
    max_pareto_configs: int = 200
    min_green_cover_fraction: float = 0.15
    min_solar_access_hours: float = 4.0
    wind_deflection_threshold_deg: float = 35.0
    downstream_radius_m: float = 400.0
    max_optimization_hours: int = 8


@dataclass
class GovernanceConfig:
    """Configuration for the Governance API."""
    host: str = "0.0.0.0"
    port: int = 8000
    api_key_header: str = "X-API-Key"
    max_response_time_s: float = 2.0
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])


@dataclass
class IngestionConfig:
    """Configuration for the data ingestion pipeline."""
    voxel_size_m: float = 2.0
    min_lidar_density: float = 20.0   # pts/m²
    max_lidar_density: float = 50.0   # pts/m²
    density_coverage_threshold: float = 0.90  # 90% of extent
    met_resample_interval_hours: float = 1.0
    kriging_station_spacing_m: float = 500.0
    min_material_signatures: int = 5000
    max_processing_hours: int = 24


@dataclass
class UpdateConfig:
    """Configuration for the rolling model update cycle."""
    retraining_interval_days: int = 90  # Quarterly
    drift_threshold_c: float = 0.4  # °C, auto-flag if exceeded
    checkpoint_retention_count: int = 12  # Keep last 3 years


@dataclass
class Config:
    """Top-level configuration for the entire pipeline."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    pinn: PINNConfig = field(default_factory=PINNConfig)
    cfd: CFDConfig = field(default_factory=CFDConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables with defaults."""
        config = cls()
        config.database.host = os.getenv("PGHOST", config.database.host)
        config.database.port = int(os.getenv("PGPORT", str(config.database.port)))
        config.database.database = os.getenv("PGDATABASE", config.database.database)
        config.database.user = os.getenv("PGUSER", config.database.user)
        config.database.password = os.getenv("PGPASSWORD", config.database.password)
        config.log_level = os.getenv("LOG_LEVEL", config.log_level)
        config.data_dir = os.getenv("DATA_DIR", config.data_dir)
        config.checkpoint_dir = os.getenv("CHECKPOINT_DIR", config.checkpoint_dir)
        return config
