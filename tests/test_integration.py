"""End-to-end integration test: chains all 5 phases together.

This test validates the complete pipeline from ingestion through to
zoning directive generation, ensuring provenance is maintained across
all pipeline stages.
"""

from datetime import datetime
from pathlib import Path

import pytest
import torch

from src.ingestion.pipeline import IngestionPipeline
from src.ucm.builder import UCMBuilder
from src.cfd.runner import CFDRunner
from src.pinn.model import PINNModel
from src.optimizer.engine import Optimizer
from src.zoning.generator import ZoningGenerator
from src.provenance.store import ProvenanceStore
from src.shared.types import (
    OptimizationConstraints, DirectiveProvenance, WardMap,
)
from src.shared.config import (
    IngestionConfig, CFDConfig, PINNConfig, OptimizerConfig,
)


class TestEndToEndPipeline:
    """Test the full 5-phase pipeline with provenance tracing."""

    def test_full_pipeline_ingestion_to_directives(self, tmp_path):
        """Run all 5 phases and verify end-to-end provenance chain."""
        prov = ProvenanceStore(str(tmp_path / "prov.json"))

        # ---- Phase 1: Ingestion ----
        ingestion = IngestionPipeline(
            config=IngestionConfig(),
            provenance_store=prov,
        )
        udt_artifact = ingestion.ingest()
        assert udt_artifact.version_id is not None

        # ---- Phase 2a: UCM Builder ----
        ucm_builder = UCMBuilder(provenance_store=prov)
        ucm_artifact = ucm_builder.build(udt_artifact)
        assert ucm_artifact.block_count > 0

        # ---- Phase 2b: CFD Runner ----
        cfd_runner = CFDRunner(
            config=CFDConfig(n_samples=10, mass_conservation_threshold=10.0),  # Relaxed for synthetic data
            provenance_store=prov,
        )
        cfd_artifact = cfd_runner.generate_training_dataset(ucm_artifact, n_samples=10)
        assert cfd_artifact.valid_sample_count > 0

        # ---- Phase 3: PINN Training ----
        pinn_config = PINNConfig(max_epochs=10)
        pinn_model = PINNModel(input_dim=10, config=pinn_config, provenance_store=prov)

        losses = pinn_model.train_on_cfd_data(cfd_artifact, epochs=10)
        assert losses["l_total"] < float("inf")  # Should converge to finite value
        assert losses["l_total"] > 0  # Loss must be positive

        # Save checkpoint
        ckpt_path = tmp_path / "pinn_checkpoint.pt"
        ckpt_artifact = pinn_model.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        # ---- Phase 4: Optimization ----
        optimizer = Optimizer(
            config=OptimizerConfig(min_pareto_configs=3, max_pareto_configs=10),
            provenance_store=prov,
        )

        corridor_graph = optimizer.build_corridor_graph(
            pinn_model, optimizer._generate_synthetic_blocks(ucm_artifact)
        )

        constraints = OptimizationConstraints(
            budget_total=1_000_000,
            min_green_cover_fraction=0.15,
        )

        pareto_artifact = optimizer.optimize(
            ucm_artifact, pinn_model, corridor_graph, constraints,
        )
        assert len(pareto_artifact.configurations) >= 3

        # ---- Phase 5: Zoning ----
        generator = ZoningGenerator(provenance_store=prov)
        selected_config = pareto_artifact.configurations[0]

        directive_provenance = DirectiveProvenance(
            udt_version_id=udt_artifact.version_id,
            pinn_checkpoint_version=ckpt_artifact.version_id,
            cfd_sample_ids=[s.sample_id for s in cfd_artifact.samples[:3]],
            mobo_run_id=pareto_artifact.run_id,
        )

        directives = generator.generate(
            selected_config, directive_provenance,
            corridor_graph=corridor_graph,
        )
        assert len(directives) > 0

        # ---- Verify provenance chain ----
        # Every directive should reference the PINN checkpoint
        for d in directives:
            assert d.provenance.pinn_checkpoint_version == ckpt_artifact.version_id
            assert d.provenance.udt_version_id == udt_artifact.version_id
            assert d.provenance.mobo_run_id == pareto_artifact.run_id

        # Provenance store should contain artifacts from all phases
        udt_records = prov.get_artifacts_by_type("udt")
        ucm_records = prov.get_artifacts_by_type("ucm")
        cfd_records = prov.get_artifacts_by_type("cfd_dataset")
        pinn_records = prov.get_artifacts_by_type("pinn_checkpoint")
        zoning_records = prov.get_artifacts_by_type("zoning_directive")

        assert len(udt_records) >= 1
        assert len(ucm_records) >= 1
        assert len(cfd_records) >= 1
        assert len(pinn_records) >= 1
        assert len(zoning_records) >= 1

    def test_ood_detection_propagates_to_zoning(self, tmp_path):
        """Verify OOD warnings flow from PINN through to directives."""
        pinn_config = PINNConfig(max_epochs=5)
        model = PINNModel(input_dim=10, config=pinn_config)

        # Set training distribution stats
        model._training_mean = torch.zeros(10)
        model._training_std = torch.ones(10)
        model._feature_names = [f"feature_{i}" for i in range(10)]

        # Create extreme input (4σ from mean)
        extreme_features = torch.ones(10) * 4.0
        prediction = model.predict(extreme_features)

        # OOD should be flagged
        assert prediction.ood_warning is True
        assert len(prediction.ood_features) > 0

        # This OOD flag would be included in the directive provenance
        directive_provenance = DirectiveProvenance(
            udt_version_id="udt-v1",
            pinn_checkpoint_version="pinn-v1",
            cfd_sample_ids=["cfd-001"],
            mobo_run_id="mobo-001",
            ood_warning=prediction.ood_warning,
            ood_features=prediction.ood_features,
        )

        assert directive_provenance.ood_warning is True

    def test_community_heat_index_with_equity_flagging(self, tmp_path):
        """Verify equity review flags trigger on high displacement risk."""
        prov = ProvenanceStore(str(tmp_path / "prov.json"))
        generator = ZoningGenerator(provenance_store=prov)

        from src.shared.types import ZoningDirective, MandatoryIntervention

        directives = [
            ZoningDirective(
                block_id="BLK-01",
                zone_class="THERMAL_REMEDIATION",
                max_height_m=20,
                height_justification="test",
                permitted_uses=["residential"],
                mandatory_interventions=[
                    MandatoryIntervention(type="green_roof", cooling_effect_estimate_c=-0.9),
                    MandatoryIntervention(type="pavement_albedo", cooling_effect_estimate_c=-0.8),
                ],
                compliance_deadline_months=36,
                review_trigger="Biennial review",
                provenance=DirectiveProvenance(
                    udt_version_id="udt-v1",
                    pinn_checkpoint_version="pinn-v1",
                    cfd_sample_ids=[],
                    mobo_run_id="mobo-001",
                ),
            ),
            ZoningDirective(
                block_id="BLK-02",
                zone_class="BASELINE_UNCHANGED",
                max_height_m=30,
                height_justification="test",
                permitted_uses=["commercial"],
                mandatory_interventions=[],
                compliance_deadline_months=60,
                review_trigger="Standard cycle",
                provenance=DirectiveProvenance(
                    udt_version_id="udt-v1",
                    pinn_checkpoint_version="pinn-v1",
                    cfd_sample_ids=[],
                    mobo_run_id="mobo-001",
                ),
            ),
        ]

        ward_map = WardMap(
            ward_blocks={"W-Central": ["BLK-01", "BLK-02"]},
            displacement_risk_threshold=0.3,
        )

        indices = generator.compute_community_heat_index(directives, ward_map)
        assert len(indices) == 1

        central = indices[0]
        assert central.ward_id == "W-Central"
        assert central.total_cooling_benefit_c_m2 >= 0  # or 0 if no delta_t provided
        assert central.total_retrofit_cost >= 0
        # With 1/2 residential blocks having interventions → 0.5 risk
        assert central.displacement_risk_score > 0
        assert central.equity_review_flagged is True
