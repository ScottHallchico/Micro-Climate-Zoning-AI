"""Initial schema: UDT artifacts, UCM blocks, directives, provenance.

Revision ID: 001_initial
Revises: -
Create Date: 2026-05-10

Creates the core PostGIS-backed tables for the Micro-Climate Zoning AI system.
All artifact tables include spatial columns for GIS integration and JSONB fields
for flexible metadata storage.
"""

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


def upgrade() -> None:
    # Enable PostGIS extension
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    # ---- UDT Artifacts ----
    op.create_table(
        "udt_artifacts",
        sa.Column("version_id", sa.String(64), primary_key=True),
        sa.Column("voxel_resolution_m", sa.Float, nullable=False, server_default="2.0"),
        sa.Column("creation_timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("data_sources", JSONB, nullable=True),
        sa.Column("bbox", sa.Text, nullable=True),  # Geometry handled via PostGIS
        sa.Column("metadata", JSONB, nullable=True),
    )

    # ---- UDT Voxels ----
    op.create_table(
        "udt_voxels",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("udt_version_id", sa.String(64), sa.ForeignKey("udt_artifacts.version_id"), nullable=False),
        sa.Column("x", sa.Float, nullable=False),
        sa.Column("y", sa.Float, nullable=False),
        sa.Column("z", sa.Float, nullable=False),
        sa.Column("geometry_type", sa.String(32), nullable=False),
        sa.Column("albedo", sa.Float, nullable=False),
        sa.Column("emissivity", sa.Float, nullable=False),
        sa.Column("heat_capacity", sa.Float, nullable=False),
        sa.Column("boundary_temp_k", sa.Float, nullable=False),
        sa.Column("material_class", sa.String(64), nullable=True),
    )
    op.create_index("ix_udt_voxels_version", "udt_voxels", ["udt_version_id"])
    op.create_index("ix_udt_voxels_coords", "udt_voxels", ["x", "y", "z"])

    # ---- UCM Blocks ----
    op.create_table(
        "ucm_blocks",
        sa.Column("block_id", sa.String(64), primary_key=True),
        sa.Column("udt_version_id", sa.String(64), sa.ForeignKey("udt_artifacts.version_id"), nullable=False),
        sa.Column("svf", sa.Float, nullable=False),
        sa.Column("lambda_p", sa.Float, nullable=False),
        sa.Column("lambda_f", sa.Float, nullable=False),
        sa.Column("hw_ratio", sa.Float, nullable=False),
        sa.Column("z0", sa.Float, nullable=False),
        sa.Column("canyon_class", sa.String(32), nullable=False),
        sa.Column("turbulence_perturbation_params", JSONB, nullable=True),
        sa.Column("building_heights", JSONB, nullable=True),
        sa.Column("block_area_m2", sa.Float, nullable=True),
        sa.Column("footprint_area_m2", sa.Float, nullable=True),
    )
    op.create_index("ix_ucm_blocks_canyon", "ucm_blocks", ["canyon_class"])

    # ---- UCM Artifacts ----
    op.create_table(
        "ucm_artifacts",
        sa.Column("version_id", sa.String(64), primary_key=True),
        sa.Column("udt_version_id", sa.String(64), sa.ForeignKey("udt_artifacts.version_id"), nullable=False),
        sa.Column("block_count", sa.Integer, nullable=False),
        sa.Column("creation_timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("canyon_class_distribution", JSONB, nullable=True),
    )

    # ---- Zoning Directives ----
    op.create_table(
        "zoning_directives",
        sa.Column("version_id", sa.String(64), primary_key=True),
        sa.Column("block_id", sa.String(64), nullable=False, index=True),
        sa.Column("zone_class", sa.String(32), nullable=False),
        sa.Column("max_height_m", sa.Float, nullable=True),
        sa.Column("height_justification", sa.Text, nullable=True),
        sa.Column("permitted_uses", JSONB, nullable=True),
        sa.Column("mandatory_interventions", JSONB, nullable=True),
        sa.Column("compliance_deadline_months", sa.Integer, nullable=True),
        sa.Column("review_trigger", sa.Text, nullable=True),
        sa.Column("provenance", JSONB, nullable=True),
        sa.Column("creation_timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("optimization_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("corridor_vector_bearing_deg", sa.Float, nullable=True),
        sa.Column("corridor_avg_wind_speed_ms", sa.Float, nullable=True),
        sa.Column("downstream_cooling_loss_c", sa.Float, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
    )
    op.create_index("ix_zoning_directives_zone", "zoning_directives", ["zone_class"])
    op.create_index("ix_zoning_directives_active", "zoning_directives", ["block_id", "is_active"])

    # ---- Provenance Records ----
    op.create_table(
        "provenance_records",
        sa.Column("artifact_id", sa.String(64), primary_key=True),
        sa.Column("artifact_type", sa.String(32), nullable=False),
        sa.Column("producing_component", sa.String(64), nullable=False),
        sa.Column("input_artifact_ids", JSONB, nullable=True),
        sa.Column("creation_timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("metadata", JSONB, nullable=True),
    )
    op.create_index("ix_provenance_type", "provenance_records", ["artifact_type"])

    # ---- Audit Log ----
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("block_id", sa.String(64), nullable=False),
        sa.Column("api_key_identifier", sa.String(64), nullable=True),
        sa.Column("compliance_outcome", sa.String(32), nullable=False),
        sa.Column("request_payload", JSONB, nullable=True),
    )
    op.create_index("ix_audit_log_block", "audit_log", ["block_id"])
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"])

    # ---- Ward Heat Index ----
    op.create_table(
        "ward_heat_index",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("ward_id", sa.String(64), nullable=False),
        sa.Column("total_cooling_benefit_c_m2", sa.Float, nullable=True),
        sa.Column("total_retrofit_cost", sa.Float, nullable=True),
        sa.Column("displacement_risk_score", sa.Float, nullable=True),
        sa.Column("equity_review_flagged", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_ward_heat_index_ward", "ward_heat_index", ["ward_id"])


def downgrade() -> None:
    op.drop_table("ward_heat_index")
    op.drop_table("audit_log")
    op.drop_table("provenance_records")
    op.drop_table("zoning_directives")
    op.drop_table("ucm_artifacts")
    op.drop_table("ucm_blocks")
    op.drop_table("udt_voxels")
    op.drop_table("udt_artifacts")
    op.execute("DROP EXTENSION IF EXISTS postgis")
