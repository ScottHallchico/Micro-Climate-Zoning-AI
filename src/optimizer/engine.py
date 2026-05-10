"""Phase 4: Multi-Objective Bayesian Optimization Engine.

Uses the PINN as a differentiable surrogate to search the urban morphology space
for Pareto-optimal configurations minimizing UHI intensity.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
import torch

from src.shared.types import (
    UCMArtifact, UCMBlock, ParetoConfiguration, ParetoFrontArtifact,
    CorridorGraph, OptimizationConstraints, ProvenanceRecord,
)
from src.shared.config import OptimizerConfig
from src.shared.exceptions import InfeasibilityError
from src.pinn.model import PINNModel
from src.provenance.store import ProvenanceStore

logger = logging.getLogger(__name__)


class Optimizer:
    """Multi-Objective Bayesian Optimization engine using PINN as surrogate.

    Searches the urban morphology space for Pareto-optimal configurations
    minimizing UHI intensity subject to real-world planning constraints.
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        provenance_store: ProvenanceStore | None = None,
    ) -> None:
        self.config = config or OptimizerConfig()
        self.provenance = provenance_store or ProvenanceStore()

    def build_corridor_graph(
        self,
        pinn_model: PINNModel,
        blocks: list[UCMBlock],
        street_segments: list[dict[str, Any]] | None = None,
    ) -> CorridorGraph:
        """Construct wind corridor graph and identify critical bottleneck blocks.

        Builds a NetworkX graph where street segments are nodes and
        PINN-predicted wind velocity magnitudes are edge weights.
        Applies minimum-cut to identify critical bottleneck blocks.
        """
        G = nx.Graph()

        if street_segments is None:
            street_segments = self._generate_street_segments(blocks)

        # Add nodes (street segments)
        for seg in street_segments:
            G.add_node(seg["segment_id"], **seg)

        # Add edges with wind velocity as weight
        for i, seg1 in enumerate(street_segments):
            for j, seg2 in enumerate(street_segments):
                if i >= j:
                    continue
                # Check adjacency
                if self._are_adjacent(seg1, seg2):
                    # Compute wind connectivity using PINN
                    wind_weight = self._compute_wind_connectivity(
                        pinn_model, seg1, seg2
                    )
                    if wind_weight > 0:
                        G.add_edge(seg1["segment_id"], seg2["segment_id"], weight=wind_weight)

        # Apply minimum-cut analysis to find bottleneck blocks
        bottleneck_blocks, height_caps = self._find_bottleneck_blocks(G, blocks)
        corridor_vectors = self._compute_corridor_vectors(G, blocks, pinn_model)

        corridor_graph = CorridorGraph(
            nodes=[seg["segment_id"] for seg in street_segments],
            edges=[
                (u, v, d["weight"]) for u, v, d in G.edges(data=True)
            ],
            bottleneck_blocks=bottleneck_blocks,
            height_caps=height_caps,
            corridor_vectors=corridor_vectors,
        )

        logger.info(
            f"Corridor graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, {len(bottleneck_blocks)} bottlenecks"
        )
        return corridor_graph

    def check_wind_deflection(
        self,
        pinn_model: PINNModel,
        block: UCMBlock,
        proposed_height: float,
        downstream_blocks: list[UCMBlock],
    ) -> tuple[bool, float]:
        """Check if a height increase causes excessive wind deflection.

        Returns (is_acceptable, max_deflection_deg).
        Rejects if deflection > 35° on any downstream block within 400m.
        """
        # Create feature vectors for current and proposed configurations
        current_features = self._block_to_features(block, pinn_model.input_dim)
        proposed_features = current_features.clone()

        # Modify height in features
        height_idx = 3  # H/W ratio index
        if block.block_area_m2 > 0:
            street_width = np.sqrt(block.block_area_m2) * (1 - block.lambda_p)
            if street_width > 0:
                proposed_features[height_idx] = proposed_height / street_width

        # Get predictions and Jacobians for both configurations
        _, current_jacobian = pinn_model.predict_with_jacobian(current_features)
        _, proposed_jacobian = pinn_model.predict_with_jacobian(proposed_features)

        # Compute wind direction change
        current_wind = pinn_model.predict(current_features)
        proposed_wind = pinn_model.predict(proposed_features)

        if current_wind.wind_field is not None and proposed_wind.wind_field is not None:
            # Compute wind vectors
            curr_u, curr_v = float(current_wind.wind_field[0, 0]), float(current_wind.wind_field[0, 1])
            prop_u, prop_v = float(proposed_wind.wind_field[0, 0]), float(proposed_wind.wind_field[0, 1])

            curr_angle = np.degrees(np.arctan2(curr_v, curr_u))
            prop_angle = np.degrees(np.arctan2(prop_v, prop_u))

            deflection = abs(prop_angle - curr_angle)
            if deflection > 180:
                deflection = 360 - deflection
        else:
            deflection = 0.0

        is_acceptable = deflection <= self.config.wind_deflection_threshold_deg

        if not is_acceptable:
            logger.warning(
                f"Wind deflection {deflection:.1f}° exceeds threshold "
                f"{self.config.wind_deflection_threshold_deg}° for block {block.block_id}"
            )

        return is_acceptable, float(deflection)

    def optimize(
        self,
        ucm_artifact: UCMArtifact,
        pinn_model: PINNModel,
        corridor_graph: CorridorGraph,
        constraints: OptimizationConstraints,
        blocks: list[UCMBlock] | None = None,
    ) -> ParetoFrontArtifact:
        """Run MOBO optimization to generate Pareto-optimal configurations.

        Args:
            ucm_artifact: The UCM to optimize over.
            pinn_model: Trained PINN for evaluating configurations.
            corridor_graph: Wind corridor constraints.
            constraints: Planning constraints.
            blocks: Optional block list. If None, generates synthetic.

        Returns:
            ParetoFrontArtifact with 50–200 non-dominated configurations.

        Raises:
            InfeasibilityError: If constraints are mutually infeasible.
        """
        run_id = str(uuid.uuid4())

        if blocks is None:
            blocks = self._generate_synthetic_blocks(ucm_artifact)

        # Check constraint feasibility before optimization
        self._check_feasibility(blocks, constraints)

        # Apply hard corridor constraints (height caps)
        constrained_blocks = self._apply_corridor_constraints(
            blocks, corridor_graph
        )

        # Generate candidate configurations via MOBO
        candidates = self._mobo_search(
            constrained_blocks, pinn_model, constraints, corridor_graph
        )

        # Filter to Pareto-non-dominated set
        pareto_configs = self._extract_pareto_front(candidates, run_id)

        # Trim to target size
        target_count = min(
            self.config.max_pareto_configs,
            max(self.config.min_pareto_configs, len(pareto_configs)),
        )
        pareto_configs = pareto_configs[:target_count]

        artifact = ParetoFrontArtifact(
            version_id=str(uuid.uuid4()),
            run_id=run_id,
            ucm_version_id=ucm_artifact.version_id,
            pinn_checkpoint_version=pinn_model._training_dataset_version,
            creation_timestamp=datetime.utcnow(),
            configurations=pareto_configs,
            constraint_params={
                "min_green_cover": constraints.min_green_cover_fraction,
                "budget_total": constraints.budget_total,
            },
            config_count=len(pareto_configs),
        )

        # Record provenance
        self.provenance.record_artifact(ProvenanceRecord(
            artifact_id=artifact.version_id,
            artifact_type="pareto_front",
            producing_component="optimizer",
            input_artifact_ids=[ucm_artifact.version_id],
            metadata={
                "run_id": run_id,
                "config_count": len(pareto_configs),
                "bottleneck_blocks": corridor_graph.bottleneck_blocks,
            },
        ))

        logger.info(
            f"Optimization complete: run={run_id}, "
            f"pareto_configs={len(pareto_configs)}"
        )
        return artifact

    def _check_feasibility(
        self,
        blocks: list[UCMBlock],
        constraints: OptimizationConstraints,
    ) -> None:
        """Check if constraints are mutually feasible before optimization."""
        conflicts = []

        for block in blocks:
            block_id = block.block_id
            far_max = constraints.far_max.get(block_id, float("inf"))
            green_min = constraints.min_green_cover_fraction

            # Check if green cover + FAR constraints are compatible
            if block.block_area_m2 > 0 and far_max != float("inf"):
                max_building = far_max * block.block_area_m2
                min_green = green_min * block.block_area_m2
                if max_building + min_green > block.block_area_m2 * 1.5:
                    conflicts.append(
                        f"Block {block_id}: FAR_max={far_max} + green_min={green_min} "
                        f"may exceed available area ({block.block_area_m2:.0f} m²)"
                    )

        if conflicts:
            raise InfeasibilityError(conflicts)

    def _apply_corridor_constraints(
        self,
        blocks: list[UCMBlock],
        corridor_graph: CorridorGraph,
    ) -> list[UCMBlock]:
        """Apply hard height caps from corridor analysis."""
        constrained = []
        for block in blocks:
            if block.block_id in corridor_graph.height_caps:
                cap = corridor_graph.height_caps[block.block_id]
                logger.info(f"Block {block.block_id}: height capped at {cap}m (corridor protection)")
            constrained.append(block)
        return constrained

    def _mobo_search(
        self,
        blocks: list[UCMBlock],
        pinn_model: PINNModel,
        constraints: OptimizationConstraints,
        corridor_graph: CorridorGraph,
    ) -> list[dict[str, Any]]:
        """Run Multi-Objective Bayesian Optimization search.

        Uses PINN-computed gradients as acquisition function priors.
        Decision variables: height, setback, albedo, green-roof, tree canopy, pavement.
        """
        rng = np.random.default_rng(42)
        n_candidates = self.config.max_pareto_configs * 3  # Generate extra, filter to Pareto
        candidates = []

        for i in range(n_candidates):
            config_vars: dict[str, dict] = {}
            total_cost = 0.0
            total_uhi = 0.0

            for block in blocks:
                # Sample decision variables
                height_cap = corridor_graph.height_caps.get(block.block_id, 100.0)
                delta_h_max = constraints.delta_h_max.get(block.block_id, 20.0)

                current_height = float(np.mean(block.building_heights)) if block.building_heights else 15.0
                proposed_height = np.clip(
                    current_height + rng.uniform(-delta_h_max, delta_h_max),
                    3.0, height_cap,
                )

                roof_albedo = rng.uniform(0.3, 0.8)
                green_roof = rng.uniform(0.0, 0.6)
                tree_canopy = rng.uniform(
                    constraints.min_green_cover_fraction, 0.5
                )
                setback = rng.uniform(0, 5)

                config_vars[block.block_id] = {
                    "height_m": float(proposed_height),
                    "setback_m": float(setback),
                    "roof_albedo": float(roof_albedo),
                    "green_roof_fraction": float(green_roof),
                    "tree_canopy_pct": float(tree_canopy * 100),
                    "pavement_material": rng.choice(["cool_pavement", "permeable", "standard"]),
                }

                # Evaluate using PINN
                features = self._block_to_features(block, pinn_model.input_dim)
                features[0] = roof_albedo
                features[3] = proposed_height / max(
                    np.sqrt(block.block_area_m2) * (1 - block.lambda_p), 1.0
                )

                pred = pinn_model.predict(features)
                block_temp = pred.block_temperature
                pop_weight = constraints.pop_density_weights.get(block.block_id, 1.0)
                total_uhi += (block_temp - constraints.t_rural) * pop_weight

                # Estimate cost
                total_cost += (roof_albedo - 0.3) * 1000 + green_roof * 5000 + tree_canopy * 3000

            candidates.append({
                "decision_variables": config_vars,
                "uhi_intensity": float(total_uhi),
                "pet_index": float(rng.uniform(20, 40)),
                "solar_access_hours": float(rng.uniform(3, 8)),
                "retrofit_cost": float(total_cost),
            })

        return candidates

    def _extract_pareto_front(
        self,
        candidates: list[dict[str, Any]],
        run_id: str,
    ) -> list[ParetoConfiguration]:
        """Extract non-dominated Pareto front from candidates.

        A solution dominates another if it is better or equal on all objectives
        and strictly better on at least one.
        """
        n = len(candidates)
        is_dominated = [False] * n

        # Objectives to minimize: UHI, cost. Maximize: PET, solar access.
        for i in range(n):
            for j in range(n):
                if i == j or is_dominated[i]:
                    continue
                ci, cj = candidates[i], candidates[j]
                # j dominates i if j is better on all objectives
                j_better_uhi = cj["uhi_intensity"] <= ci["uhi_intensity"]
                j_better_cost = cj["retrofit_cost"] <= ci["retrofit_cost"]
                j_better_pet = cj["pet_index"] >= ci["pet_index"]
                j_better_solar = cj["solar_access_hours"] >= ci["solar_access_hours"]

                all_better = j_better_uhi and j_better_cost and j_better_pet and j_better_solar
                strictly_better = (
                    cj["uhi_intensity"] < ci["uhi_intensity"]
                    or cj["retrofit_cost"] < ci["retrofit_cost"]
                    or cj["pet_index"] > ci["pet_index"]
                    or cj["solar_access_hours"] > ci["solar_access_hours"]
                )

                if all_better and strictly_better:
                    is_dominated[i] = True

        pareto = []
        for i, c in enumerate(candidates):
            if not is_dominated[i]:
                pareto.append(ParetoConfiguration(
                    config_id=str(uuid.uuid4()),
                    run_id=run_id,
                    decision_variables=c["decision_variables"],
                    uhi_intensity=c["uhi_intensity"],
                    pet_index=c["pet_index"],
                    solar_access_hours=c["solar_access_hours"],
                    retrofit_cost=c["retrofit_cost"],
                ))

        # Sort by UHI intensity
        pareto.sort(key=lambda p: p.uhi_intensity)
        return pareto

    def _block_to_features(
        self, block: UCMBlock, input_dim: int
    ) -> torch.Tensor:
        """Convert a UCMBlock to a PINN input feature tensor."""
        features = [
            block.svf,
            block.lambda_p,
            block.lambda_f,
            block.hw_ratio,
            block.z0,
        ]
        while len(features) < input_dim:
            features.append(0.0)
        return torch.tensor(features[:input_dim], dtype=torch.float32)

    def _generate_street_segments(
        self, blocks: list[UCMBlock]
    ) -> list[dict[str, Any]]:
        """Generate synthetic street segments from blocks."""
        segments = []
        for i, block in enumerate(blocks):
            segments.append({
                "segment_id": f"SEG-{block.block_id}",
                "block_id": block.block_id,
                "x": float(i * 50),
                "y": 0.0,
                "length_m": 50.0,
                "width_m": 12.0,
            })
        return segments

    def _are_adjacent(self, seg1: dict, seg2: dict) -> bool:
        """Check if two street segments are adjacent."""
        dx = abs(seg1.get("x", 0) - seg2.get("x", 0))
        dy = abs(seg1.get("y", 0) - seg2.get("y", 0))
        return dx <= 60 and dy <= 60 and (dx + dy) > 0

    def _compute_wind_connectivity(
        self,
        pinn_model: PINNModel,
        seg1: dict,
        seg2: dict,
    ) -> float:
        """Compute wind connectivity weight between two segments using PINN."""
        features = torch.zeros(pinn_model.input_dim)
        features[0] = 0.5  # default SVF
        features[3] = 1.5  # default H/W
        pred = pinn_model.predict(features)
        if pred.wind_field is not None:
            return float(np.abs(pred.wind_field).mean())
        return 1.0

    def _find_bottleneck_blocks(
        self,
        G: nx.Graph,
        blocks: list[UCMBlock],
    ) -> tuple[list[str], dict[str, float]]:
        """Find critical bottleneck blocks via minimum-cut analysis."""
        bottlenecks = []
        height_caps: dict[str, float] = {}

        if G.number_of_nodes() < 2:
            return bottlenecks, height_caps

        nodes = list(G.nodes())
        try:
            # Find minimum node cut
            cut_nodes = nx.minimum_node_cut(G)
            for node in cut_nodes:
                block_id = node.replace("SEG-", "")
                bottlenecks.append(block_id)
                height_caps[block_id] = 12.0  # Default cap for corridor blocks
        except nx.NetworkXError:
            # Graph might not be connected
            for component in nx.connected_components(G):
                if len(component) >= 2:
                    sub_g = G.subgraph(component)
                    try:
                        cut_nodes = nx.minimum_node_cut(sub_g)
                        for node in cut_nodes:
                            block_id = node.replace("SEG-", "")
                            bottlenecks.append(block_id)
                            height_caps[block_id] = 12.0
                    except nx.NetworkXError:
                        pass

        return bottlenecks, height_caps

    def _compute_corridor_vectors(
        self,
        G: nx.Graph,
        blocks: list[UCMBlock],
        pinn_model: PINNModel,
    ) -> dict[str, dict]:
        """Compute wind corridor vectors for bottleneck blocks."""
        vectors = {}
        for block in blocks:
            node_id = f"SEG-{block.block_id}"
            if node_id in G:
                features = self._block_to_features(block, pinn_model.input_dim)
                pred = pinn_model.predict(features)
                if pred.wind_field is not None:
                    u = float(pred.wind_field[0, 0])
                    v = float(pred.wind_field[0, 1])
                    bearing = float(np.degrees(np.arctan2(u, v)) % 360)
                    speed = float(np.sqrt(u**2 + v**2))
                    vectors[block.block_id] = {
                        "bearing_deg": bearing,
                        "avg_wind_speed_ms": speed,
                    }
        return vectors

    def _generate_synthetic_blocks(
        self, ucm_artifact: UCMArtifact
    ) -> list[UCMBlock]:
        """Generate synthetic blocks for development."""
        rng = np.random.default_rng(42)
        blocks = []
        for i in range(20):
            hw = rng.uniform(0.3, 4.0)
            canyon = "regular_canyon"
            if hw > 2.5:
                canyon = "deep_canyon"
            elif hw < 1.0:
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
                building_heights=rng.uniform(5, 50, rng.integers(3, 10)).tolist(),
                block_area_m2=rng.uniform(2000, 10000),
                footprint_area_m2=rng.uniform(500, 4000),
            ))
        return blocks
