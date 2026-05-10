"""Tests for Optimizer (Phase 4) and Zoning Generator (Phase 5)."""

import json
from pathlib import Path

import pytest

from src.optimizer.engine import Optimizer
from src.zoning.generator import ZoningGenerator
from src.pinn.model import PINNModel
from src.shared.types import (
    UCMArtifact, UCMBlock, CorridorGraph, OptimizationConstraints,
    ParetoConfiguration, DirectiveProvenance, WardMap, ZoningDirective,
    MandatoryIntervention,
)
from src.shared.config import OptimizerConfig, PINNConfig
from src.shared.exceptions import InfeasibilityError
from src.provenance.store import ProvenanceStore


class TestOptimizer:
    """Test corridor graph, wind deflection, and MOBO optimization."""

    def _make_optimizer(self, tmp_path: Path) -> Optimizer:
        return Optimizer(
            config=OptimizerConfig(min_pareto_configs=5, max_pareto_configs=20),
            provenance_store=ProvenanceStore(str(tmp_path / "prov.json")),
        )

    def _make_pinn(self) -> PINNModel:
        return PINNModel(input_dim=10, config=PINNConfig(max_epochs=2))

    def test_build_corridor_graph(self, tmp_path):
        opt = self._make_optimizer(tmp_path)
        pinn = self._make_pinn()
        blocks = self._make_blocks()

        graph = opt.build_corridor_graph(pinn, blocks)
        assert len(graph.nodes) > 0
        assert len(graph.edges) >= 0

    def test_infeasibility_error_raised(self, tmp_path):
        opt = self._make_optimizer(tmp_path)
        blocks = [UCMBlock(
            block_id="BLK-01", udt_version_id="v1",
            svf=0.5, lambda_p=0.3, lambda_f=0.2,
            hw_ratio=1.5, z0=0.5, canyon_class="regular_canyon",
            block_area_m2=100,  # Very small area
        )]
        constraints = OptimizationConstraints(
            far_max={"BLK-01": 10.0},  # High FAR
            min_green_cover_fraction=0.9,  # 90% green cover
        )
        with pytest.raises(InfeasibilityError):
            opt._check_feasibility(blocks, constraints)

    def test_pareto_front_non_dominated(self, tmp_path):
        opt = self._make_optimizer(tmp_path)
        pinn = self._make_pinn()
        ucm = UCMArtifact()
        blocks = self._make_blocks()
        graph = CorridorGraph()
        constraints = OptimizationConstraints(budget_total=1e9)

        artifact = opt.optimize(ucm, pinn, graph, constraints, blocks)
        configs = artifact.configurations

        # Verify no solution dominates another in the Pareto front
        for i, ci in enumerate(configs):
            for j, cj in enumerate(configs):
                if i == j:
                    continue
                # cj should NOT dominate ci
                dominates = (
                    cj.uhi_intensity <= ci.uhi_intensity
                    and cj.retrofit_cost <= ci.retrofit_cost
                    and cj.pet_index >= ci.pet_index
                    and cj.solar_access_hours >= ci.solar_access_hours
                    and (
                        cj.uhi_intensity < ci.uhi_intensity
                        or cj.retrofit_cost < ci.retrofit_cost
                        or cj.pet_index > ci.pet_index
                        or cj.solar_access_hours > ci.solar_access_hours
                    )
                )
                assert not dominates, f"Config {j} dominates config {i} in Pareto front"

    def test_height_caps_on_bottlenecks(self, tmp_path):
        opt = self._make_optimizer(tmp_path)
        pinn = self._make_pinn()
        blocks = self._make_blocks()

        graph = opt.build_corridor_graph(pinn, blocks)
        # Bottleneck blocks should have height caps
        for block_id in graph.bottleneck_blocks:
            assert block_id in graph.height_caps

    def _make_blocks(self, n: int = 5) -> list[UCMBlock]:
        return [
            UCMBlock(
                block_id=f"BLK-{i + 1:02d}",
                udt_version_id="v1",
                svf=0.5 + i * 0.05,
                lambda_p=0.3,
                lambda_f=0.2,
                hw_ratio=1.0 + i * 0.3,
                z0=0.5,
                canyon_class="regular_canyon",
                building_heights=[10.0 + i * 5],
                block_area_m2=5000.0,
                footprint_area_m2=2000.0,
            )
            for i in range(n)
        ]


class TestZoningGenerator:
    """Test zoning directive generation, schema validation, and round-trip."""

    def _make_generator(self, tmp_path: Path) -> ZoningGenerator:
        return ZoningGenerator(
            provenance_store=ProvenanceStore(str(tmp_path / "prov.json"))
        )

    def _make_provenance(self) -> DirectiveProvenance:
        return DirectiveProvenance(
            udt_version_id="udt-v1",
            pinn_checkpoint_version="pinn-v1",
            cfd_sample_ids=["cfd-001", "cfd-002"],
            mobo_run_id="mobo-run-001",
        )

    def test_generate_produces_directives(self, tmp_path):
        gen = self._make_generator(tmp_path)
        config = ParetoConfiguration(
            decision_variables={
                "BLK-01": {"height_m": 20, "roof_albedo": 0.5,
                           "green_roof_fraction": 0.1, "tree_canopy_pct": 20},
                "BLK-02": {"height_m": 15, "roof_albedo": 0.3,
                           "green_roof_fraction": 0.0, "tree_canopy_pct": 10},
            },
        )
        directives = gen.generate(config, self._make_provenance())
        assert len(directives) == 2
        assert all(d.block_id in ("BLK-01", "BLK-02") for d in directives)

    def test_each_block_has_exactly_one_zone_class(self, tmp_path):
        gen = self._make_generator(tmp_path)
        config = ParetoConfiguration(
            decision_variables={
                f"BLK-{i:02d}": {"height_m": 20, "roof_albedo": 0.5,
                                 "green_roof_fraction": 0.2, "tree_canopy_pct": 15}
                for i in range(1, 6)
            },
        )
        directives = gen.generate(config, self._make_provenance())
        block_classes = {}
        for d in directives:
            assert d.block_id not in block_classes, f"Block {d.block_id} has multiple zone classes"
            block_classes[d.block_id] = d.zone_class

    def test_wind_corridor_directive_has_corridor_fields(self, tmp_path):
        gen = self._make_generator(tmp_path)
        corridor = CorridorGraph(
            bottleneck_blocks=["BLK-01"],
            height_caps={"BLK-01": 12.0},
            corridor_vectors={"BLK-01": {"bearing_deg": 87, "avg_wind_speed_ms": 3.2}},
        )
        config = ParetoConfiguration(
            decision_variables={
                "BLK-01": {"height_m": 12, "roof_albedo": 0.3,
                           "green_roof_fraction": 0.0, "tree_canopy_pct": 10},
            },
        )
        directives = gen.generate(config, self._make_provenance(), corridor_graph=corridor)
        d = directives[0]
        assert d.zone_class == "WIND_CORRIDOR_CRITICAL"
        assert d.corridor_vector_bearing_deg is not None
        assert d.corridor_avg_wind_speed_ms is not None
        assert d.downstream_cooling_loss_c is not None

    def test_height_justification_references_ids(self, tmp_path):
        gen = self._make_generator(tmp_path)
        prov = self._make_provenance()
        config = ParetoConfiguration(
            decision_variables={
                "BLK-01": {"height_m": 20, "roof_albedo": 0.3,
                           "green_roof_fraction": 0.0, "tree_canopy_pct": 10},
            },
        )
        directives = gen.generate(config, prov)
        d = directives[0]
        assert prov.pinn_checkpoint_version in d.height_justification
        assert prov.mobo_run_id in d.height_justification

    def test_json_round_trip_fidelity(self, tmp_path):
        """Property 2: parse(serialize(directive)) == directive (field-by-field)."""
        gen = self._make_generator(tmp_path)
        directive = ZoningDirective(
            block_id="BLK-09",
            zone_class="THERMAL_REMEDIATION",
            max_height_m=None,
            height_justification="Test justification",
            permitted_uses=["residential", "commercial"],
            mandatory_interventions=[
                MandatoryIntervention(
                    type="green_roof",
                    cooling_effect_estimate_c=-0.9,
                    min_coverage_pct=60.0,
                    species="sedum_mosaic",
                ),
            ],
            compliance_deadline_months=36,
            review_trigger="Biennial review",
            provenance=self._make_provenance(),
        )

        serialized = ZoningGenerator.serialize_directive(directive)
        json_str = json.dumps(serialized)
        parsed_dict = json.loads(json_str)
        parsed = ZoningGenerator.parse_directive(parsed_dict)

        assert parsed.block_id == directive.block_id
        assert parsed.zone_class == directive.zone_class
        assert parsed.max_height_m == directive.max_height_m
        assert parsed.compliance_deadline_months == directive.compliance_deadline_months
        assert len(parsed.mandatory_interventions) == len(directive.mandatory_interventions)
        assert parsed.provenance.mobo_run_id == directive.provenance.mobo_run_id

    def test_community_heat_index_displacement_flagging(self, tmp_path):
        gen = self._make_generator(tmp_path)
        directives = [
            ZoningDirective(
                block_id="BLK-01", zone_class="THERMAL_REMEDIATION",
                max_height_m=20, height_justification="test",
                permitted_uses=["residential"],
                mandatory_interventions=[
                    MandatoryIntervention(type="green_roof", cooling_effect_estimate_c=-0.5),
                ],
                compliance_deadline_months=36,
                review_trigger="test",
                provenance=self._make_provenance(),
            ),
        ]
        ward_map = WardMap(
            ward_blocks={"W1": ["BLK-01"]},
            displacement_risk_threshold=0.3,
        )

        indices = gen.compute_community_heat_index(directives, ward_map)
        assert len(indices) == 1
        # Single residential block with interventions → displacement risk = 1.0
        assert indices[0].displacement_risk_score > 0
        assert indices[0].equity_review_flagged is True

    def test_schema_validation_rejects_missing_field(self, tmp_path):
        gen = self._make_generator(tmp_path)
        import jsonschema
        from src.shared.schemas import ZONING_DIRECTIVE_SCHEMA

        incomplete = {"block_id": "BLK-01"}  # Missing required fields
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(incomplete, ZONING_DIRECTIVE_SCHEMA)
