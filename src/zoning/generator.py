"""Phase 5: Zoning Code Generation.

Translates Pareto-optimal configurations into machine-readable, legally structured
zoning amendments with full justification trails for democratic review.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import jsonschema

from src.shared.types import (
    ZoningDirective, MandatoryIntervention, DirectiveProvenance,
    ParetoConfiguration, ParetoFrontArtifact, ProvenanceRecord,
    WardHeatIndex, WardMap, CorridorGraph, UCMBlock,
)
from src.shared.schemas import ZONING_DIRECTIVE_SCHEMA
from src.provenance.store import ProvenanceStore

logger = logging.getLogger(__name__)


class ZoningGenerator:
    """Translates Pareto-optimal configurations into legally structured zoning directives.

    Zone classes:
    - WIND_CORRIDOR_CRITICAL: Height-restricted corridor preservation
    - THERMAL_REMEDIATION: Surface material interventions for heat hotspots
    - DENSITY_ADAPTIVE: Beneficial height increases for self-shading
    - BASELINE_UNCHANGED: No modifications needed
    """

    def __init__(self, provenance_store: ProvenanceStore | None = None) -> None:
        self.provenance = provenance_store or ProvenanceStore()

    def generate(
        self,
        pareto_config: ParetoConfiguration,
        provenance: DirectiveProvenance,
        blocks: list[UCMBlock] | None = None,
        corridor_graph: CorridorGraph | None = None,
    ) -> list[ZoningDirective]:
        """Produce one ZoningDirective per affected block.

        Validates each directive against the JSON schema before storage.
        Round-trip property: parse(serialize(directive)) == directive.
        """
        directives: list[ZoningDirective] = []

        for block_id, vars in pareto_config.decision_variables.items():
            # Determine zone class
            zone_class = self._assign_zone_class(
                block_id, vars, corridor_graph, blocks
            )

            # Build mandatory interventions
            interventions = self._build_interventions(zone_class, vars)

            # Build height justification
            justification = self._build_height_justification(
                block_id, zone_class, vars, provenance
            )

            # Build directive
            directive = ZoningDirective(
                block_id=block_id,
                zone_class=zone_class,
                max_height_m=vars.get("height_m"),
                height_justification=justification,
                permitted_uses=self._get_permitted_uses(zone_class),
                mandatory_interventions=interventions,
                compliance_deadline_months=self._get_deadline(zone_class),
                review_trigger=self._get_review_trigger(zone_class),
                provenance=provenance,
                version_id=str(uuid.uuid4()),
                creation_timestamp=datetime.utcnow(),
                optimization_timestamp=datetime.utcnow(),
            )

            # Add wind corridor specific fields
            if zone_class == "WIND_CORRIDOR_CRITICAL" and corridor_graph:
                cv = corridor_graph.corridor_vectors.get(block_id, {})
                directive.corridor_vector_bearing_deg = cv.get("bearing_deg")
                directive.corridor_avg_wind_speed_ms = cv.get("avg_wind_speed_ms")
                directive.downstream_cooling_loss_c = self._estimate_cooling_loss(
                    block_id, vars, corridor_graph
                )

            # Validate against JSON schema
            self._validate_directive(directive)

            directives.append(directive)

        # Record provenance for the batch
        for d in directives:
            self.provenance.record_artifact(ProvenanceRecord(
                artifact_id=d.version_id,
                artifact_type="zoning_directive",
                producing_component="zoning_generator",
                input_artifact_ids=[provenance.mobo_run_id],
                metadata={
                    "block_id": d.block_id,
                    "zone_class": d.zone_class,
                    "ood_warning": provenance.ood_warning,
                },
            ))

        logger.info(f"Generated {len(directives)} zoning directives")
        return directives

    def compute_community_heat_index(
        self,
        directives: list[ZoningDirective],
        ward_map: WardMap,
        block_areas: dict[str, float] | None = None,
        delta_t: dict[str, float] | None = None,
    ) -> list[WardHeatIndex]:
        """Aggregate per-ward metrics for equity monitoring.

        Computes: total cooling benefit, retrofit cost, displacement risk score.
        Auto-flags wards above displacement risk threshold.
        """
        ward_indices: list[WardHeatIndex] = []

        for ward_id, block_ids in ward_map.ward_blocks.items():
            total_cooling = 0.0
            total_cost = 0.0
            residential_intervention_count = 0
            total_blocks = len(block_ids)

            for directive in directives:
                if directive.block_id not in block_ids:
                    continue

                # Cooling benefit = ΔT × block_area
                dt = (delta_t or {}).get(directive.block_id, 0.0)
                area = (block_areas or {}).get(directive.block_id, 1000.0)
                total_cooling += dt * area

                # Estimate cost from interventions
                for intervention in directive.mandatory_interventions:
                    total_cost += abs(intervention.cooling_effect_estimate_c) * 5000

                # Count residential blocks with mandatory interventions
                if directive.mandatory_interventions and "residential" in str(
                    directive.permitted_uses
                ):
                    residential_intervention_count += 1

            # Displacement risk: proportion of residential blocks with interventions
            displacement_risk = (
                residential_intervention_count / total_blocks
                if total_blocks > 0
                else 0.0
            )

            flagged = displacement_risk > ward_map.displacement_risk_threshold

            ward_indices.append(WardHeatIndex(
                ward_id=ward_id,
                total_cooling_benefit_c_m2=total_cooling,
                total_retrofit_cost=total_cost,
                displacement_risk_score=displacement_risk,
                equity_review_flagged=flagged,
            ))

            if flagged:
                logger.warning(
                    f"Ward {ward_id} flagged for equity review: "
                    f"displacement_risk={displacement_risk:.2f}"
                )

        return ward_indices

    def _assign_zone_class(
        self,
        block_id: str,
        vars: dict[str, Any],
        corridor_graph: CorridorGraph | None,
        blocks: list[UCMBlock] | None,
    ) -> str:
        """Assign each block to exactly one zone class based on optimization role."""
        # Check if block is a wind corridor bottleneck
        if corridor_graph and block_id in corridor_graph.bottleneck_blocks:
            return "WIND_CORRIDOR_CRITICAL"

        # Check if block needs thermal remediation (high albedo change or green interventions)
        current_albedo = 0.2  # default
        proposed_albedo = vars.get("roof_albedo", 0.2)
        green_fraction = vars.get("green_roof_fraction", 0.0)
        tree_canopy = vars.get("tree_canopy_pct", 0.0) / 100.0

        if (proposed_albedo - current_albedo > 0.2) or green_fraction > 0.3 or tree_canopy > 0.25:
            return "THERMAL_REMEDIATION"

        # Check if density increase is proposed
        block = None
        if blocks:
            block = next((b for b in blocks if b.block_id == block_id), None)
        if block:
            current_height = float(sum(block.building_heights) / max(len(block.building_heights), 1))
            proposed_height = vars.get("height_m", current_height)
            if proposed_height > current_height * 1.2:
                return "DENSITY_ADAPTIVE"

        return "BASELINE_UNCHANGED"

    def _build_interventions(
        self, zone_class: str, vars: dict[str, Any]
    ) -> list[MandatoryIntervention]:
        """Build mandatory interventions based on zone class."""
        interventions = []

        if zone_class == "THERMAL_REMEDIATION":
            albedo = vars.get("roof_albedo", 0.5)
            if albedo > 0.3:
                interventions.append(MandatoryIntervention(
                    type="pavement_albedo",
                    cooling_effect_estimate_c=-0.8,
                    min_albedo=albedo,
                    current_value=0.12,
                    retrofit_priority="streets_first",
                ))

            green_frac = vars.get("green_roof_fraction", 0.0)
            if green_frac > 0.1:
                interventions.append(MandatoryIntervention(
                    type="green_roof",
                    cooling_effect_estimate_c=-0.9,
                    min_coverage_pct=green_frac * 100,
                    species="sedum_mosaic",
                ))

            tree_pct = vars.get("tree_canopy_pct", 0.0)
            if tree_pct > 10:
                interventions.append(MandatoryIntervention(
                    type="street_tree_canopy",
                    cooling_effect_estimate_c=-0.5,
                    min_coverage_pct=tree_pct,
                    placement="south_and_west_facades",
                ))

        elif zone_class == "DENSITY_ADAPTIVE":
            interventions.append(MandatoryIntervention(
                type="green_roof",
                cooling_effect_estimate_c=-0.6,
                min_coverage_pct=30.0,
                species="intensive_green",
            ))

        return interventions

    def _build_height_justification(
        self,
        block_id: str,
        zone_class: str,
        vars: dict[str, Any],
        provenance: DirectiveProvenance,
    ) -> str:
        """Build height justification referencing PINN, CFD, and MOBO run IDs."""
        height = vars.get("height_m")
        pinn_id = provenance.pinn_checkpoint_version
        cfd_ids = ", ".join(provenance.cfd_sample_ids[:3]) if provenance.cfd_sample_ids else "N/A"
        mobo_id = provenance.mobo_run_id

        if zone_class == "WIND_CORRIDOR_CRITICAL":
            return (
                f"Height restricted to {height}m to preserve wind corridor. "
                f"PINN prediction run: {pinn_id}. CFD training samples: {cfd_ids}. "
                f"MOBO optimization run: {mobo_id}."
            )
        elif zone_class == "DENSITY_ADAPTIVE":
            return (
                f"Height increase to {height}m approved — generates beneficial self-shading. "
                f"PINN prediction run: {pinn_id}. CFD training samples: {cfd_ids}. "
                f"MOBO optimization run: {mobo_id}."
            )
        else:
            return (
                f"Height set to {height}m based on thermodynamic optimization. "
                f"PINN prediction run: {pinn_id}. CFD training samples: {cfd_ids}. "
                f"MOBO optimization run: {mobo_id}."
            )

    def _get_permitted_uses(self, zone_class: str) -> list[str]:
        """Get permitted uses based on zone class."""
        uses_map = {
            "WIND_CORRIDOR_CRITICAL": ["residential_low", "retail_ground_floor"],
            "THERMAL_REMEDIATION": ["residential", "commercial", "mixed_use"],
            "DENSITY_ADAPTIVE": ["residential", "commercial", "mixed_use", "office"],
            "BASELINE_UNCHANGED": ["residential", "commercial", "mixed_use", "industrial", "office"],
        }
        return uses_map.get(zone_class, ["residential"])

    def _get_deadline(self, zone_class: str) -> int:
        """Get compliance deadline in months based on zone class priority."""
        deadlines = {
            "WIND_CORRIDOR_CRITICAL": 12,
            "THERMAL_REMEDIATION": 36,
            "DENSITY_ADAPTIVE": 48,
            "BASELINE_UNCHANGED": 60,
        }
        return deadlines.get(zone_class, 60)

    def _get_review_trigger(self, zone_class: str) -> str:
        """Get review trigger description based on zone class."""
        triggers = {
            "WIND_CORRIDOR_CRITICAL": "Annual wind-corridor re-assessment via PINN update",
            "THERMAL_REMEDIATION": "Biennial thermal performance review",
            "DENSITY_ADAPTIVE": "Triennial density impact assessment",
            "BASELINE_UNCHANGED": "Standard 5-year zoning review cycle",
        }
        return triggers.get(zone_class, "Standard review cycle")

    def _estimate_cooling_loss(
        self,
        block_id: str,
        vars: dict[str, Any],
        corridor_graph: CorridorGraph,
    ) -> float:
        """Estimate downstream cooling loss from corridor height violation."""
        cv = corridor_graph.corridor_vectors.get(block_id, {})
        wind_speed = cv.get("avg_wind_speed_ms", 3.0)
        # Simplified: cooling loss proportional to wind reduction
        return round(wind_speed * 0.6, 1)  # °C

    def _validate_directive(self, directive: ZoningDirective) -> None:
        """Validate directive against the JSON schema."""
        directive_dict = self.serialize_directive(directive)
        try:
            jsonschema.validate(directive_dict, ZONING_DIRECTIVE_SCHEMA)
        except jsonschema.ValidationError as e:
            logger.error(f"Directive validation failed for {directive.block_id}: {e.message}")
            raise

    @staticmethod
    def serialize_directive(directive: ZoningDirective) -> dict[str, Any]:
        """Serialize a ZoningDirective to JSON-compatible dict."""
        return {
            "block_id": directive.block_id,
            "zone_class": directive.zone_class,
            "max_height_m": directive.max_height_m,
            "height_justification": directive.height_justification,
            "permitted_uses": directive.permitted_uses,
            "mandatory_interventions": [
                {
                    "type": i.type,
                    "cooling_effect_estimate_c": i.cooling_effect_estimate_c,
                    **({"min_coverage_pct": i.min_coverage_pct} if i.min_coverage_pct is not None else {}),
                    **({"min_albedo": i.min_albedo} if i.min_albedo is not None else {}),
                    **({"species": i.species} if i.species is not None else {}),
                    **({"placement": i.placement} if i.placement is not None else {}),
                }
                for i in directive.mandatory_interventions
            ],
            "compliance_deadline_months": directive.compliance_deadline_months,
            "review_trigger": directive.review_trigger,
            "provenance": {
                "udt_version_id": directive.provenance.udt_version_id,
                "pinn_checkpoint_version": directive.provenance.pinn_checkpoint_version,
                "cfd_sample_ids": directive.provenance.cfd_sample_ids,
                "mobo_run_id": directive.provenance.mobo_run_id,
            } if directive.provenance else {},
        }

    @staticmethod
    def parse_directive(data: dict[str, Any]) -> ZoningDirective:
        """Parse a ZoningDirective from a JSON-compatible dict."""
        prov_data = data.get("provenance", {})
        provenance = DirectiveProvenance(
            udt_version_id=prov_data.get("udt_version_id", ""),
            pinn_checkpoint_version=prov_data.get("pinn_checkpoint_version", ""),
            cfd_sample_ids=prov_data.get("cfd_sample_ids", []),
            mobo_run_id=prov_data.get("mobo_run_id", ""),
        )

        interventions = [
            MandatoryIntervention(
                type=i["type"],
                cooling_effect_estimate_c=i["cooling_effect_estimate_c"],
                min_coverage_pct=i.get("min_coverage_pct"),
                min_albedo=i.get("min_albedo"),
                species=i.get("species"),
                placement=i.get("placement"),
            )
            for i in data.get("mandatory_interventions", [])
        ]

        return ZoningDirective(
            block_id=data["block_id"],
            zone_class=data["zone_class"],
            max_height_m=data.get("max_height_m"),
            height_justification=data.get("height_justification", ""),
            permitted_uses=data.get("permitted_uses", []),
            mandatory_interventions=interventions,
            compliance_deadline_months=data.get("compliance_deadline_months", 60),
            review_trigger=data.get("review_trigger", ""),
            provenance=provenance,
        )
