# Implementation Plan: Micro-Climate Zoning AI

## Overview

Implement the five-phase pipeline in dependency order: data ingestion → UCM construction → CFD orchestration → PINN training and serialization → optimization → zoning generation → governance API → scenario dashboard. A central provenance store and rolling update cycle are wired in throughout. Each phase produces versioned artifacts consumed by the next.

## Tasks

- [ ] 1. Set up project structure, shared types, and provenance store
  - Create the top-level package layout: `ingestion/`, `ucm/`, `cfd/`, `pinn/`, `optimizer/`, `zoning/`, `governance/`, `dashboard/`, `provenance/`, `shared/`
  - Define shared Python dataclasses and type aliases: `UDTArtifact`, `UCMArtifact`, `CFDDatasetArtifact`, `CFDSample`, `CheckpointArtifact`, `CheckpointManifest`, `ParetoFrontArtifact`, `ParetoConfiguration`, `ProvenanceRecord`, `WardHeatIndex`, `PermitApplication`, `ComplianceResponse`
  - Implement `ProvenanceStore` with `record_artifact(record: ProvenanceRecord) -> None` and `get_lineage(artifact_id: str) -> list[ProvenanceRecord]` backed by a PostGIS table
  - Set up PostGIS schema migrations (Alembic) for all artifact tables: `udt_artifacts`, `ucm_blocks`, `cfd_samples`, `pinn_checkpoints`, `pareto_fronts`, `zoning_directives`, `provenance_records`, `audit_log`
  - Configure pytest, hypothesis, and project-level `pyproject.toml`
  - _Requirements: 13.4_

- [ ] 2. Implement Ingestion_Pipeline
  - [ ] 2.1 Implement LIDAR ingestion: CSF ground classification via PDAL, alpha-shape building polygon extrusion via Open3D, LoD-2 mesh generation
    - Validate point density 20–50 pts/m² for ≥90% of spatial extent; log structured warning per sub-region below threshold
    - _Requirements: 1.1, 15.1_

  - [ ] 2.2 Implement satellite thermal ingestion: homography co-registration to LIDAR extent, NDVI × NDWI cloud-masking
    - Validate atmospheric emissivity correction metadata presence; hard-reject image and log structured error if absent
    - _Requirements: 1.2, 15.2_

  - [ ] 2.3 Implement meteorological stream ingestion: resample to 1-hour timestep, kriging interpolation for zones with station spacing > 500m
    - _Requirements: 1.3_

  - [ ] 2.4 Implement material classification: Random Forest on ECOSTRESS Spectral Library (≥5,000 signatures), assign α, ε, ρCp per surface
    - _Requirements: 1.4_

  - [ ] 2.5 Implement UDT voxelization at 2m × 2m × 2m, fusing all preprocessed streams into `UDTVoxel` records; persist to PostGIS in CityGML/3DCityDB-compatible format; assign unique `version_id`; record provenance artifact
    - Implement `IngestionPipeline.ingest()` returning `UDTArtifact`; raise `IngestionValidationError` on hard failures; log structured warnings on partial stream failures
    - _Requirements: 1.5, 1.6, 1.7, 13.3, 13.4_

  - [ ] 2.6 Write unit tests for Ingestion_Pipeline
    - Test CSF classification output shape and ground/non-ground label distribution
    - Test emissivity metadata rejection path (missing metadata → `IngestionValidationError`)
    - Test kriging interpolation produces values within meteorological plausible range
    - Test material classifier assigns α ∈ [0,1], ε ∈ [0,1], ρCp > 0 for all surfaces
    - Test partial stream failure: pipeline continues and logs structured warning
    - _Requirements: 1.6, 15.1, 15.2_

- [ ] 3. Implement UCM_Builder
  - [ ] 3.1 Implement morphological index computation for each city block: SVF, λp, λf, H/W, z₀
    - Validate input fields: building height ∈ [0, 600] m, block area > 0 m²; raise on invalid
    - _Requirements: 2.1, 15.3_

  - [ ] 3.2 Implement canyon classification logic: deep_canyon (H/W > 2.5), regular_canyon (1 ≤ H/W ≤ 2.5), shallow_canyon (H/W < 1), transitional_zone (variable H/W between adjacent blocks)
    - Assign stochastic turbulence perturbation parameters to all transitional_zone blocks
    - _Requirements: 2.2, 2.3_

  - [ ] 3.3 Implement `UCMBuilder.build()`: persist all `UCMBlock` records to PostGIS linked to UDT block identifiers; return `UCMArtifact` with `version_id`; record provenance artifact
    - _Requirements: 2.4, 13.4_

  - [ ] 3.4 Write unit tests for UCM_Builder
    - Test each canyon class boundary (H/W = 0.9, 1.0, 2.5, 2.6) maps to correct class
    - Test transitional_zone blocks receive non-null turbulence perturbation params
    - Test invalid height (< 0, > 600) raises validation error
    - Test invalid block area (≤ 0) raises validation error
    - _Requirements: 2.2, 2.3, 15.3_

- [ ] 4. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement CFD_Runner
  - [ ] 5.1 Implement stratified sample selection: 200 samples via Latin Hypercube Sampling over morphological index space, ensuring coverage across all four canyon classification types
    - _Requirements: 3.1_

  - [ ] 5.2 Implement OpenFOAM RANS k-ε orchestration: generate case files from `UCMBlock` parameters, submit runs, collect wind fields (u, v, w, P) and energy fields (Qh, Qe, ΔQs, Q*) at voxel resolution
    - _Requirements: 3.2_

  - [ ] 5.3 Implement mass conservation validation: compute ∇·u = 0 residual; accept if < 1×10⁻⁴; log failure with sample_id and morphology params and flag for re-run if not
    - _Requirements: 3.4, 3.5_

  - [ ] 5.4 Implement `CFDRunner.generate_training_dataset()`: assemble validated `CFDSample` records into a versioned `CFDDatasetArtifact`; persist to PostGIS; record provenance artifact
    - _Requirements: 3.6, 13.4_

  - [ ] 5.5 Write unit tests for CFD_Runner
    - Test LHS sampling produces exactly 200 samples with representation from all four canyon classes
    - Test mass conservation check: residual < 1×10⁻⁴ → status "valid"; residual ≥ 1×10⁻⁴ → status "failed" with log entry
    - Test failed samples are excluded from the returned `CFDDatasetArtifact`
    - _Requirements: 3.1, 3.4, 3.5_

- [ ] 6. Implement PINN_Model architecture and training
  - [ ] 6.1 Implement shared SIREN encoder: 8-layer MLP, 512 neurons/layer, sin activations, using PyTorch/DeepXDE
    - _Requirements: 4.1_

  - [ ] 6.2 Implement wind head (outputs u, v, w, P per voxel) and energy head (outputs Qh, Qe, ΔQs, Q* per voxel) as separate `nn.Module` subclasses sharing the encoder
    - _Requirements: 4.1_

  - [ ] 6.3 Implement composite physics loss: L_total = L_data + λ₁·L_NS + λ₂·L_energy + λ₃·L_BC
    - L_NS: Navier-Stokes continuity and momentum residuals including buoyancy coupling term ρg·β(T − T∞)
    - L_energy: urban surface energy balance Q* = Qh + Qe + ΔQs
    - L_BC: no-slip at building walls, logarithmic wind profile at domain top, periodic lateral BCs
    - _Requirements: 4.2_

  - [ ] 6.4 Implement `PINNModel.predict()`: run forward pass, attach OOD warning flag if any feature is beyond 3σ from training distribution mean
    - _Requirements: 4.4, 15.4_

  - [ ] 6.5 Implement `PINNModel.predict_with_jacobian()`: return prediction and ∂output/∂input Jacobian via `torch.autograd.functional.jacobian`
    - _Requirements: 4.6_

  - [ ] 6.6 Implement fine-tuning support: `PINNModel.fine_tune()` freezes encoder weights, updates only final 2 layers of each head using provided `WeatherMeasurement` list
    - _Requirements: 4.3, 4.7_

  - [ ] 6.7 Implement multi-scenario inference: diurnal cycle (24h × 15-min steps), seasonal variation (4 canonical days), extreme heat events (90th-percentile days), future climate projections (RCP 4.5 and RCP 8.5)
    - _Requirements: 4.5_

  - [ ] 6.8 Write unit tests for PINN_Model
    - Test forward pass output shapes match expected voxel grid dimensions for wind and energy heads
    - Test OOD flag is set when a feature exceeds 3σ from training mean
    - Test OOD flag is not set for in-distribution inputs
    - Test fine_tune() does not modify encoder weights (compare parameter checksums before/after)
    - Test predict() completes in < 0.3 s on inference hardware (performance assertion)
    - _Requirements: 4.4, 15.4_

- [ ] 7. Implement PINN checkpoint serialization with round-trip fidelity
  - [ ] 7.1 Implement `PINNModel.save_checkpoint()`: serialize architecture config, weights, λ₁/λ₂/λ₃, training dataset version, fine-tuning city ID, and timestamp to a versioned file; return `CheckpointArtifact`; record provenance artifact
    - _Requirements: 5.1, 5.3, 13.4_

  - [ ] 7.2 Implement `PINNModel.load_checkpoint()`: deserialize and validate checkpoint; raise `CheckpointError` with descriptive message on corruption or architecture mismatch; never return a partially initialized model
    - _Requirements: 5.4_

  - [ ] 7.3 Write property test for PINN checkpoint round-trip fidelity
    - **Property 1: PINN checkpoint round-trip**
    - For any trained `PINNModel` and any in-distribution input tensor, `load_checkpoint(save_checkpoint(model)).predict(x)` produces predictions numerically identical (within float32 epsilon) to `model.predict(x)` before serialization
    - **Validates: Requirements 5.2**

  - [ ] 7.4 Write unit tests for checkpoint serialization
    - Test `CheckpointManifest` fields are all present and correctly typed after save
    - Test loading a truncated/corrupt file raises `CheckpointError` (not a partial model)
    - Test loading a checkpoint with mismatched architecture config raises `CheckpointError`
    - Test version_id includes training dataset version, city ID, and timestamp components
    - _Requirements: 5.3, 5.4_

- [ ] 8. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement Optimizer — Corridor_Graph and MOBO engine
  - [ ] 9.1 Implement `Optimizer.build_corridor_graph()`: construct NetworkX graph where street segments are nodes and PINN-predicted wind velocity magnitudes are edge weights; apply minimum-cut analysis to identify critical bottleneck blocks; assign hard height caps to bottleneck blocks
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ] 9.2 Implement wind deflection check for transitional zone proposals: compute PINN Jacobian for proposed height increase; reject if downstream wind deflection > 35° on any block within 400m
    - _Requirements: 6.4_

  - [ ] 9.3 Implement infeasibility detection: before MOBO search, check all hard constraints for mutual feasibility (FAR, ΔH, wind corridor, green cover, budget); raise `InfeasibilityError` with conflicting constraint details if infeasible
    - _Requirements: 7.2, 15.5_

  - [ ] 9.4 Implement BoTorch MOBO search over decision variables (building heights, setbacks, roof albedo, green-roof fraction, street tree canopy %, pavement material) with PINN Jacobians as acquisition function priors
    - Primary objective: minimize UHI intensity Σ_blocks [T_block(x) − T_rural] · Pop_density_weight
    - Secondary objectives: maximize PET index, preserve solar access ≥ 4h/day, minimize retrofit cost
    - _Requirements: 7.1, 7.3, 7.4, 7.6_

  - [ ] 9.5 Implement `Optimizer.optimize()`: return `ParetoFrontArtifact` with 50–200 non-dominated configurations; persist each configuration with run_id, input constraints, and full objective values to PostGIS; record provenance artifact
    - _Requirements: 7.5, 7.7, 13.4_

  - [ ] 9.6 Write unit tests for Optimizer
    - Test `build_corridor_graph()` produces a graph with correct node count matching street segment count
    - Test minimum-cut identifies the expected bottleneck block in a synthetic 3-node graph
    - Test hard height caps on bottleneck blocks are not overridden by MOBO proposals
    - Test wind deflection > 35° causes proposal rejection; ≤ 35° is accepted
    - Test `InfeasibilityError` is raised when green cover + FAR constraints exceed block area
    - Test Pareto front contains only non-dominated solutions (no solution dominates another)
    - _Requirements: 6.3, 6.4, 7.5, 15.5_

- [ ] 10. Implement Zoning_Generator
  - [ ] 10.1 Implement zone class assignment: map each block's optimization role to exactly one of WIND_CORRIDOR_CRITICAL, THERMAL_REMEDIATION, DENSITY_ADAPTIVE, BASELINE_UNCHANGED
    - _Requirements: 8.2_

  - [ ] 10.2 Implement `ZoningGenerator.generate()`: produce one `ZoningDirective` JSON per affected block containing all required fields (block_id, zone_class, max_height_m, height_justification referencing PINN run ID + CFD sample ID + MOBO run ID, permitted_uses, mandatory_interventions with cooling_effect_estimate_c, compliance_deadline_months, review_trigger, provenance)
    - For WIND_CORRIDOR_CRITICAL blocks, include corridor vector (bearing °), average wind speed (m/s), and downstream cooling loss (°C) in the directive
    - Validate each directive against the JSON schema before storage; persist to PostGIS with version_id and optimization timestamp; record provenance artifact
    - _Requirements: 8.1, 8.3, 8.4, 8.5, 8.6, 6.5, 13.1, 13.4_

  - [ ] 10.3 Implement `ZoningGenerator.compute_community_heat_index()`: aggregate per-ward metrics (total cooling benefit = Σ ΔT × block_area, total retrofit cost, displacement risk score); auto-flag wards above displacement risk threshold for equity review
    - _Requirements: 11.1, 11.2_

  - [ ] 10.4 Write property test for zoning directive round-trip fidelity
    - **Property 2: Zoning directive JSON round-trip**
    - For any valid `ZoningDirective` object, `parse(serialize(directive)) == directive` (field-by-field equality after JSON serialization and deserialization)
    - **Validates: Requirements 8.7**

  - [ ] 10.5 Write unit tests for Zoning_Generator
    - Test each zone class is assigned to exactly one block (no block has two classes)
    - Test WIND_CORRIDOR_CRITICAL directive contains corridor_vector, avg_wind_speed, downstream_cooling_loss fields
    - Test height_justification references PINN run ID, CFD sample ID, and MOBO run ID
    - Test schema validation rejects a directive missing a required field
    - Test community heat index displacement_risk_score > threshold sets equity_review_flagged = True
    - _Requirements: 8.1, 8.2, 8.4, 8.7, 11.2_

- [ ] 11. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement Governance_API
  - [ ] 12.1 Implement FastAPI application skeleton: configure API key authentication middleware (header `X-API-Key`); return HTTP 401 for unauthenticated requests
    - _Requirements: 9.5_

  - [ ] 12.2 Implement `POST /v1/permit/check` endpoint: validate `PermitApplication` payload (HTTP 422 with per-field errors on malformed input); look up active zoning directive for `block_id` (HTTP 404 if not found); evaluate compliance against directive fields; return `ComplianceResponse` with status, violated_fields, and active_directive_version_id
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ] 12.3 Implement `GET /v1/directive/{directive_version_id}/provenance` endpoint: return full `ProvenanceRecord` for the directive including links to source data artifacts
    - _Requirements: 13.2_

  - [ ] 12.4 Implement `GET /v1/health` endpoint
    - _Requirements: 9.1_

  - [ ] 12.5 Implement audit logging middleware: log every permit check request with timestamp, block_id, API key identifier (not value), and compliance outcome to the `audit_log` PostGIS table
    - _Requirements: 9.6_

  - [ ] 12.6 Write unit tests for Governance_API
    - Test unauthenticated request returns HTTP 401
    - Test unknown block_id returns HTTP 404 with descriptive message
    - Test malformed payload (missing required field) returns HTTP 422 with per-field errors
    - Test compliant permit application returns status "compliant" and empty violated_fields
    - Test non-compliant application returns status "non_compliant" and lists violated fields
    - Test audit log entry is written for every permit check (including rejected ones)
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ] 12.7 Write integration test for Governance_API performance
    - Simulate 100 concurrent permit check requests; assert p95 latency < 2 s
    - _Requirements: 9.2, 14.4_

- [ ] 13. Implement Scenario_Dashboard
  - [ ] 13.1 Implement React/Deck.gl block map: render all city blocks color-coded by zone class and UHI intensity; wire to Governance_API for directive data
    - _Requirements: 10.1_

  - [ ] 13.2 Implement planner override flow: capture directive parameter change (height, albedo, green cover) → call PINN inference endpoint → display ΔT on all affected blocks; complete within 5 s of override submission
    - Show wind corridor violation warning (affected corridor, predicted deflection angle, downstream blocks losing cooling benefit) when override would breach a hard constraint
    - _Requirements: 10.2, 10.5_

  - [ ] 13.3 Implement Pareto front selector panel: display 50–200 configurations as a selectable list; on selection, update all block-level directives on the map
    - _Requirements: 10.3_

  - [ ] 13.4 Implement community heat index panel: display per-ward cooling benefit (°C), retrofit cost burden, and displacement risk flag for the currently selected configuration; show all wards simultaneously
    - _Requirements: 10.4, 11.3_

  - [ ] 13.5 Write unit tests for Scenario_Dashboard components
    - Test block map renders correct color for each zone class
    - Test override submission triggers PINN call and updates ΔT display within 5 s (mock PINN endpoint)
    - Test wind corridor violation warning appears when override exceeds hard constraint
    - Test Pareto front selection updates all block directives on the map
    - Test community heat index panel shows equity_review_flagged indicator for flagged wards
    - _Requirements: 10.2, 10.5, 11.3_

- [ ] 14. Implement rolling update cycle and drift detection
  - [ ] 14.1 Implement quarterly retraining workflow: ingest new AWS meteorological data and satellite thermal imagery since previous cycle; fine-tune PINN without full retraining from scratch; save new versioned checkpoint; retain all previous checkpoints
    - _Requirements: 12.1, 12.4_

  - [ ] 14.2 Implement drift detection: after each retraining cycle, compute MAE on held-out validation set; compare to previous cycle baseline; if MAE on any block exceeds 0.4°C, auto-flag all active directives for that block for human review with drift magnitude annotation
    - _Requirements: 12.2, 12.3_

  - [ ] 14.3 Write unit tests for rolling update cycle
    - Test quarterly retraining produces a new checkpoint with a distinct version_id
    - Test encoder weights are unchanged after fine-tuning (frozen encoder assertion)
    - Test drift detection: MAE > 0.4°C on a block → directives for that block are flagged with drift annotation
    - Test drift detection: MAE ≤ 0.4°C → no directives flagged
    - Test all previous checkpoints are retained and loadable after a new cycle
    - _Requirements: 12.1, 12.3, 12.4_

- [ ] 15. Implement OOD detection integration across pipeline
  - [ ] 15.1 Integrate OOD detection into `PINNModel.predict()` and `predict_with_jacobian()`: compute per-feature z-scores against training distribution statistics; attach `ood_warning: bool` and `ood_features: list[str]` to every `PINNPrediction`
    - _Requirements: 15.4_

  - [ ] 15.2 Propagate OOD warnings through Optimizer and Zoning_Generator: any directive produced from an OOD prediction SHALL include an `ood_warning` annotation in its provenance record
    - _Requirements: 15.4, 13.1_

  - [ ] 15.3 Write unit tests for OOD detection
    - Test feature at exactly 3σ does not trigger OOD flag (boundary condition)
    - Test feature at 3σ + ε triggers OOD flag
    - Test OOD annotation propagates to the zoning directive provenance record
    - _Requirements: 15.4_

- [ ] 16. Final checkpoint — Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at the end of each major phase
- Property tests (7.3, 10.4) validate universal round-trip correctness properties defined in the design
- Unit tests validate specific examples, edge cases, and error conditions
- The PINN checkpoint round-trip property (7.3) is a hard requirement per Requirements 5.2
- The zoning directive JSON round-trip property (10.4) is a hard requirement per Requirements 8.7
- Performance targets (0.3 s PINN inference, 24 h LIDAR ingestion, 8 h optimizer, p95 < 2 s API) are validated in unit/integration tests where possible and must be verified on designated hardware
