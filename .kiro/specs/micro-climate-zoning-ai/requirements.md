# Requirements Document

## Introduction

The Micro-Climate Zoning AI system is a five-phase pipeline that transforms raw urban sensor data into physics-grounded, block-level zoning legislation. The system treats the city as a living thermodynamic system, fusing LIDAR geometry, satellite thermal imagery, meteorological streams, and material property libraries into a voxelized Urban Digital Twin (UDT), then applying Physics-Informed Neural Networks (PINNs) and Multi-Objective Bayesian Optimization (MOBO) to generate legally structured zoning amendments that reduce Urban Heat Island (UHI) intensity while preserving wind corridors, green cover, and democratic governance.

The system is designed to make the physics of urban heat visible and auditable — not to replace human planners or elected bodies, who retain full legal authority over all zoning decisions.

---

## Glossary

- **UDT (Urban Digital Twin)**: A 2m × 2m × 2m voxelized grid of the city where each cell carries geometry, material properties, and boundary thermal conditions. The substrate for all downstream computation.
- **UCM (Urban Canopy Model)**: A parameterized representation of a city block that governs how the block exchanges heat and momentum with the atmosphere above it.
- **PINN (Physics-Informed Neural Network)**: A neural network that simultaneously learns from CFD training data and respects Navier-Stokes and energy transport equations as hard constraints in its loss function.
- **CFD (Computational Fluid Dynamics)**: Numerical simulation of fluid flow and heat transfer. Used here via OpenFOAM RANS k-ε to generate ground-truth training data for the PINN.
- **MOBO (Multi-Objective Bayesian Optimization)**: An optimization algorithm that searches the urban morphology space for Pareto-optimal configurations minimizing UHI intensity subject to real-world constraints.
- **UHI (Urban Heat Island)**: The phenomenon where urban areas are significantly warmer than surrounding rural areas due to human activity and built-environment characteristics.
- **SVF (Sky View Factor)**: The fraction of visible sky from street level; a key morphological index governing direct radiation load.
- **FAR (Floor Area Ratio)**: The ratio of total floor area to the area of the plot; a standard zoning constraint.
- **PET (Physiological Equivalent Temperature)**: A thermal comfort index for pedestrians.
- **AWS (Automatic Weather Station)**: Ground-based meteorological measurement stations.
- **LIDAR**: Light Detection and Ranging; airborne sensor producing 3D point clouds of urban geometry.
- **SIREN**: Sinusoidal Representation Networks; neural network architecture using sin(·) activations with well-defined higher-order derivatives, required for stable PDE residual computation.
- **Pareto Front**: The set of non-dominated solutions in a multi-objective optimization, where no objective can be improved without worsening another.
- **Ingestion_Pipeline**: The Phase 1 software component responsible for acquiring, preprocessing, and fusing all raw data streams into the UDT.
- **UCM_Builder**: The Phase 2 software component that computes morphological indices and canyon classifications from the UDT.
- **CFD_Runner**: The Phase 2 software component that orchestrates OpenFOAM RANS k-ε simulations to generate PINN training data.
- **PINN_Model**: The Phase 3 multi-head neural network that predicts wind and energy fields for any block configuration.
- **Optimizer**: The Phase 4 MOBO engine that uses the PINN as a differentiable surrogate to generate Pareto-optimal zoning configurations.
- **Zoning_Generator**: The Phase 5 component that translates optimizer outputs into machine-readable, legally structured zoning directives.
- **Governance_API**: The Phase 5 FastAPI REST service through which architects and developers check building permit applications against current zoning codes.
- **Scenario_Dashboard**: The Phase 5 React/Deck.gl interactive map interface for city planners.
- **Corridor_Graph**: The NetworkX-based graph where street segments are nodes and airflow connectivity is the edge weight, used to identify and protect dominant wind vectors.

---

## Requirements

### Requirement 1: Multi-Source Data Ingestion

**User Story:** As an urban data engineer, I want to ingest LIDAR point clouds, satellite thermal imagery, meteorological station streams, and material property libraries, so that all raw urban data is fused into a single, consistent Urban Digital Twin.

#### Acceptance Criteria

1. WHEN a LIDAR point cloud dataset (20–50 pts/m²) is provided, THE Ingestion_Pipeline SHALL classify ground points using the Cloth Simulation Filter (CSF) algorithm and extrude building polygons via alpha-shape reconstruction to produce LoD-2 3D meshes.
2. WHEN satellite thermal imagery from ECOSTRESS or Landsat 8–9 thermal bands is provided, THE Ingestion_Pipeline SHALL co-register the imagery to the LIDAR spatial extent using homography transforms and apply cloud-masking via NDVI × NDWI band combinations.
3. WHEN meteorological station data is provided at 5-minute intervals, THE Ingestion_Pipeline SHALL resample all streams to a unified 1-hour timestep, applying kriging interpolation for zones with station density below 500m spacing.
4. WHEN surface spectral signatures are available, THE Ingestion_Pipeline SHALL classify roof and pavement surface materials using a Random Forest model trained on the ECOSTRESS Spectral Library (minimum 5,000 material signatures), assigning albedo (α), thermal emissivity (ε), and volumetric heat capacity (ρCp) to each surface.
5. WHEN all data streams have been preprocessed, THE Ingestion_Pipeline SHALL produce a UDT as a 2m × 2m × 2m voxelized grid where each cell carries geometry, material properties, and boundary thermal conditions.
6. IF any required data stream is unavailable or fails validation, THEN THE Ingestion_Pipeline SHALL log a structured error identifying the stream, the failure reason, and the affected spatial extent, and SHALL continue processing available streams.
7. THE Ingestion_Pipeline SHALL store the UDT in a CityGML/3DCityDB-compatible format queryable via PostGIS.

---

### Requirement 2: Urban Canopy Model Construction

**User Story:** As a climate modeller, I want each city block parameterized into a UCM with morphological indices, so that I have a thermodynamic baseline for CFD simulation and PINN training.

#### Acceptance Criteria

1. WHEN a city block is present in the UDT, THE UCM_Builder SHALL compute the following morphological indices for that block: Sky View Factor (SVF), Plan Area Fraction (λp), Frontal Area Index (λf), Mean Aspect Ratio (H/W), and aerodynamic roughness length (z₀).
2. WHEN the H/W ratio of a block is computed, THE UCM_Builder SHALL classify the block's canyon type according to the following schema: H/W > 2.5 as "deep canyon", 1 ≤ H/W ≤ 2.5 as "regular canyon", H/W < 1 as "shallow canyon", and variable H/W between adjacent blocks as "transitional zone".
3. THE UCM_Builder SHALL assign stochastic turbulence perturbation parameters to all blocks classified as "transitional zone".
4. WHEN UCM construction is complete, THE UCM_Builder SHALL persist all morphological indices and canyon classifications to the PostGIS spatial database, linked to their corresponding UDT block identifiers.

---

### Requirement 3: CFD Training Data Generation

**User Story:** As a machine learning engineer, I want ground-truth CFD simulation results for 200 stratified samples of the city morphology space, so that the PINN has physics-accurate training data.

#### Acceptance Criteria

1. WHEN CFD training data generation is initiated, THE CFD_Runner SHALL select 200 stratified samples from the city morphology space, ensuring coverage across all four canyon classification types.
2. WHEN a morphology sample is selected, THE CFD_Runner SHALL execute an OpenFOAM RANS k-ε simulation for that sample, producing wind velocity fields (u, v, w), pressure fields (P), and surface energy balance fields (Qh, Qe, ΔQs, Q*) at voxel resolution.
3. THE CFD_Runner SHALL complete each simulation run within 72 hours of wall-clock time on the designated compute infrastructure.
4. WHEN a CFD simulation completes, THE CFD_Runner SHALL validate the output by checking mass conservation (∇·u = 0 residual below 1×10⁻⁴) and SHALL store the validated results in the training dataset with a unique sample identifier.
5. IF a CFD simulation fails or produces results that fail mass conservation validation, THEN THE CFD_Runner SHALL log the failure with the sample identifier and morphology parameters, and SHALL flag the sample for re-run or replacement.
6. THE CFD_Runner SHALL expose the completed training dataset as a versioned artifact accessible to the PINN_Model training pipeline.

---

### Requirement 4: PINN Architecture and Training

**User Story:** As a machine learning engineer, I want a Physics-Informed Neural Network that learns from CFD data while respecting Navier-Stokes and energy balance equations, so that the model produces physically consistent predictions at inference speed.

#### Acceptance Criteria

1. THE PINN_Model SHALL implement a multi-head architecture with a shared encoder (8-layer MLP, 512 neurons per layer, SIREN activations) and two prediction heads: a wind head outputting (u, v, w, P) and an energy head outputting (Qh, Qe, ΔQs, Q*) at every voxel.
2. WHEN the PINN_Model is trained, THE PINN_Model SHALL minimize a composite loss function L_total = L_data + λ₁·L_NS + λ₂·L_energy + λ₃·L_BC, where L_NS enforces Navier-Stokes continuity and momentum residuals (including the buoyancy coupling term ρg·β(T − T∞)), L_energy enforces the urban surface energy balance (Q* = Qh + Qe + ΔQs), and L_BC enforces no-slip conditions at building walls, a logarithmic wind profile at the domain top, and periodic lateral boundary conditions.
3. WHEN pre-training on CFD samples is complete, THE PINN_Model SHALL support fine-tuning of only the final 2 layers of each prediction head using 10–15 local weather station measurements, without modifying the shared encoder weights.
4. WHEN given a block's geometric and material feature vector as input, THE PINN_Model SHALL produce a complete wind and energy field prediction in under 0.3 seconds per block configuration on the designated inference hardware.
5. THE PINN_Model SHALL support multi-scenario inference across the following scenario types: diurnal cycle (24 hours at 15-minute steps), seasonal variation (4 canonical days), extreme heat events (90th-percentile temperature days), and future climate projections (RCP 4.5 and RCP 8.5).
6. THE PINN_Model SHALL expose a differentiable inference interface that returns Jacobians (∂output/∂input) for use by the Optimizer as gradient priors.
7. WHEN a new city deployment is initiated, THE PINN_Model SHALL accept a pre-trained checkpoint from a prior city and fine-tune using that city's local weather station measurements, achieving fine-tuning convergence within 15 weather station measurement samples.

---

### Requirement 5: PINN Serialization and Round-Trip Fidelity

**User Story:** As an ML operations engineer, I want to serialize and deserialize PINN model checkpoints reliably, so that trained models can be stored, versioned, and reloaded without loss of predictive fidelity.

#### Acceptance Criteria

1. THE PINN_Model SHALL serialize its full state (architecture configuration, weights, training metadata, and physics constraint coefficients λ₁, λ₂, λ₃) to a checkpoint file.
2. WHEN a checkpoint file is loaded, THE PINN_Model SHALL produce predictions identical to those produced before serialization for all inputs within the training distribution (round-trip property: predictions after load(save(model)) equal predictions before save).
3. THE PINN_Model SHALL version each checkpoint with a unique identifier that includes the training dataset version, fine-tuning city identifier, and training timestamp.
4. IF a checkpoint file is corrupt or incompatible with the current model architecture version, THEN THE PINN_Model SHALL return a descriptive error identifying the incompatibility and SHALL not load a partially initialized model.

---

### Requirement 6: Wind Corridor Identification and Protection

**User Story:** As an urban planner, I want dominant wind corridors identified and protected as hard constraints in the optimization, so that cooling airflow is preserved across the city regardless of other development pressures.

#### Acceptance Criteria

1. WHEN PINN wind field predictions are available for the city, THE Optimizer SHALL construct a Corridor_Graph using NetworkX where street segments are nodes and airflow connectivity (derived from PINN-predicted wind velocity magnitudes) is the edge weight.
2. WHEN the Corridor_Graph is constructed, THE Optimizer SHALL apply minimum-cut analysis to identify critical bottleneck blocks whose removal would reduce citywide airflow connectivity below acceptable thresholds.
3. THE Optimizer SHALL assign hard height caps to all blocks identified as critical wind corridor bottlenecks, and these caps SHALL NOT be overridden by any other optimization objective.
4. WHEN a proposed building height increase is evaluated for a transitional zone block, THE Optimizer SHALL compute the PINN Jacobian to quantify downstream wind deflection before accepting the proposal, and SHALL reject any proposal that causes wind deflection greater than 35° on any downstream block within 400m.
5. WHEN a wind corridor constraint is applied to a block, THE Zoning_Generator SHALL include in the generated directive the corridor vector (bearing in degrees), average wind speed (m/s), and the quantified downstream cooling loss (°C) that would result from a height violation.

---

### Requirement 7: Thermodynamic Optimization

**User Story:** As a city planning department, I want a multi-objective optimizer that generates a Pareto front of urban configurations minimizing UHI intensity subject to planning constraints, so that planners can choose from a range of evidence-grounded options.

#### Acceptance Criteria

1. WHEN optimization is initiated for a city, THE Optimizer SHALL minimize UHI intensity defined as Σ_blocks [T_block(x) − T_rural] · Pop_density_weight across all decision variables x.
2. THE Optimizer SHALL enforce the following hard constraints during optimization: FAR_block ≤ FAR_max per block, ΔH_block ≤ ΔH_permitted per block, Wind_corridor[k] ≥ V_min for all identified corridors k, green cover ≥ 15% of block area per block, and Σ_blocks cost(x) ≤ budget_total.
3. THE Optimizer SHALL optimize the following decision variables: building heights, building setbacks, roof albedo targets, green-roof fraction, street tree canopy percentage, and pavement material selection.
4. THE Optimizer SHALL simultaneously optimize the following secondary objectives: maximize pedestrian thermal comfort (PET index), preserve solar access ≥ 4 hours per day per block, and minimize retrofit cost per block.
5. WHEN optimization completes, THE Optimizer SHALL produce a Pareto front of 50–200 non-dominated urban configurations, each with quantified values for all primary and secondary objectives.
6. THE Optimizer SHALL use PINN-computed gradients (Jacobians) as acquisition function priors within the Multi-Objective Bayesian Optimization algorithm, reducing the number of PINN evaluations required to converge.
7. WHEN the Optimizer produces a Pareto front, THE Optimizer SHALL store each configuration with a unique run identifier, the input constraint parameters, and the full set of objective values, in the PostGIS spatial database.

---

### Requirement 8: Zoning Code Generation

**User Story:** As an urban planner, I want optimizer outputs translated into machine-readable, legally structured zoning directives with full justification trails, so that the AI's recommendations are auditable and suitable for democratic review.

#### Acceptance Criteria

1. WHEN a Pareto-optimal configuration is selected, THE Zoning_Generator SHALL produce a zoning directive for each affected block in a structured JSON format containing: block_id, zone_class, max_height_m, height_justification, permitted_uses, mandatory_interventions, compliance_deadline_months, and review_trigger.
2. THE Zoning_Generator SHALL assign each block to exactly one of the following zone classes based on its optimization role: WIND_CORRIDOR_CRITICAL, THERMAL_REMEDIATION, DENSITY_ADAPTIVE, or BASELINE_UNCHANGED.
3. WHEN a mandatory intervention is specified for a block, THE Zoning_Generator SHALL include in the directive the intervention type, minimum coverage or specification, and the estimated cooling effect in °C.
4. WHEN a height restriction is imposed on a block, THE Zoning_Generator SHALL include a height_justification field that references the specific PINN prediction run identifier, the CFD training sample identifier, and the optimization run identifier that produced the restriction.
5. THE Zoning_Generator SHALL produce a machine-readable zoning amendment document that links every directive to its PINN prediction, CFD training sample, and optimization run, providing a complete auditability trail.
6. WHEN a zoning directive is generated, THE Zoning_Generator SHALL store it in the PostGIS spatial database with a version identifier and the timestamp of the optimization run that produced it.
7. THE Zoning_Generator SHALL parse and validate all generated JSON directives against the zoning directive schema before storage, and FOR ALL valid directive objects, serializing then parsing then serializing SHALL produce an equivalent object (round-trip property).

---

### Requirement 9: Governance API

**User Story:** As an architect or developer, I want a REST API to check building permit applications against current zoning codes in real time, so that I can verify compliance before formal submission.

#### Acceptance Criteria

1. THE Governance_API SHALL expose a REST endpoint that accepts a building permit application payload (block_id, proposed_height_m, proposed_FAR, proposed_roof_albedo, proposed_green_cover_pct) and returns the compliance status against the current active zoning directive for that block.
2. WHEN a permit application is submitted to the Governance_API, THE Governance_API SHALL return a response within 2 seconds containing: compliance status (compliant / non-compliant / conditional), the specific directive fields that are violated (if any), and the active directive version identifier.
3. IF a permit application references a block_id that does not exist in the PostGIS spatial database, THEN THE Governance_API SHALL return an HTTP 404 response with a descriptive error message identifying the unknown block_id.
4. IF the Governance_API receives a malformed request payload, THEN THE Governance_API SHALL return an HTTP 422 response with a structured validation error identifying each invalid field and the expected format.
5. THE Governance_API SHALL authenticate all requests using API key authentication, and SHALL reject unauthenticated requests with an HTTP 401 response.
6. THE Governance_API SHALL log all permit check requests with timestamp, block_id, applicant API key identifier (not the key value), and compliance outcome, to a persistent audit log.

---

### Requirement 10: Scenario Dashboard

**User Story:** As a city planning department member, I want an interactive map where I can override any zoning directive and immediately see the PINN-predicted thermal consequence on all affected blocks, so that I can explore trade-offs before committing to a zoning decision.

#### Acceptance Criteria

1. THE Scenario_Dashboard SHALL render all city blocks on an interactive map using Deck.gl, color-coded by zone class and current UHI intensity.
2. WHEN a planner overrides a directive parameter (height, albedo, green cover) for a block in the Scenario_Dashboard, THE Scenario_Dashboard SHALL invoke the PINN_Model and display the predicted thermal consequence (ΔT in °C) on all affected blocks within 5 seconds of the override being submitted.
3. THE Scenario_Dashboard SHALL display the Pareto front of optimizer configurations as a selectable list, and WHEN a configuration is selected, THE Scenario_Dashboard SHALL update the map to reflect all block-level directives for that configuration.
4. THE Scenario_Dashboard SHALL display a community heat index panel showing per-ward aggregate values for: cooling benefit (°C), estimated retrofit cost burden, and displacement risk flag, for the currently selected configuration.
5. WHEN a planner's override would violate a hard wind corridor constraint, THE Scenario_Dashboard SHALL display a warning identifying the affected corridor, the predicted deflection angle, and the downstream blocks that would lose cooling benefit.

---

### Requirement 11: Community Heat Index and Equity Monitoring

**User Story:** As an elected official or member of the public, I want per-ward aggregate metrics on cooling benefit, cost burden, and displacement risk, so that equity concerns are surfaced automatically before any zoning amendment is adopted.

#### Acceptance Criteria

1. THE Zoning_Generator SHALL compute a community heat index for each ward, aggregating: total cooling benefit (sum of ΔT × block area), total estimated retrofit cost, and a displacement risk score based on the proportion of mandatory interventions affecting residential blocks.
2. WHEN the community heat index for any ward shows a displacement risk score above the configured threshold, THE Zoning_Generator SHALL automatically flag that ward's directives for equity review before the amendment package is finalized.
3. THE Scenario_Dashboard SHALL display the community heat index for all wards simultaneously, enabling visual comparison of equity distribution across the city.

---

### Requirement 12: Rolling Model Update Cycle

**User Story:** As an AI operations engineer, I want the PINN to be retrained quarterly with new sensor data and zoning codes auto-flagged when model drift exceeds tolerance, so that the system's predictions remain accurate as the city evolves.

#### Acceptance Criteria

1. THE PINN_Model SHALL support a quarterly retraining workflow that ingests new AWS meteorological data and satellite thermal imagery collected since the previous training cycle, without requiring a full retraining from scratch.
2. WHEN a quarterly retraining cycle completes, THE PINN_Model SHALL compute the mean absolute prediction error on a held-out validation set and compare it to the previous cycle's baseline.
3. WHEN the mean absolute prediction error on any block exceeds 0.4°C compared to the previous model version, THE Zoning_Generator SHALL automatically flag all active zoning directives for that block for human review, annotating each directive with the magnitude of model drift detected.
4. THE PINN_Model SHALL retain all previous model checkpoints with their version identifiers, enabling rollback to any prior version within the retention period.

---

### Requirement 13: Data Provenance and Auditability

**User Story:** As an urban planner or legal reviewer, I want every zoning directive to be traceable back to its source data, model predictions, and optimization run, so that the system's recommendations can be audited and challenged in democratic or legal proceedings.

#### Acceptance Criteria

1. THE Zoning_Generator SHALL embed in every zoning directive a provenance record containing: the UDT version identifier, the PINN checkpoint version used for prediction, the CFD training sample identifiers that contributed to the relevant model weights, and the MOBO optimization run identifier.
2. THE Governance_API SHALL expose a read-only endpoint that accepts a directive version identifier and returns the full provenance record for that directive, including links to the source data artifacts.
3. THE Ingestion_Pipeline SHALL assign a unique version identifier to each UDT produced, recording the source dataset identifiers, preprocessing parameters, and ingestion timestamp.
4. WHEN any component in the pipeline produces an output artifact (UDT, CFD result, PINN checkpoint, optimization run, zoning directive), THE component SHALL record the artifact's version identifier, producing component, input artifact identifiers, and creation timestamp in a central provenance store.

---

### Requirement 14: System Performance and Scalability

**User Story:** As a system operator, I want the pipeline to handle city-scale datasets and support real-time planner interactions, so that the system is practical for deployment in large urban areas.

#### Acceptance Criteria

1. THE Ingestion_Pipeline SHALL process a full city LIDAR dataset of up to 500 km² within 24 hours on the designated compute infrastructure.
2. THE PINN_Model SHALL produce wind and energy field predictions for a single block configuration in under 0.3 seconds on the designated inference hardware.
3. THE Optimizer SHALL produce a Pareto front of 50–200 configurations for a city of up to 10,000 blocks within 8 hours of wall-clock time.
4. THE Governance_API SHALL support a minimum of 100 concurrent permit check requests with a p95 response latency below 2 seconds under normal operating conditions.
5. WHILE the Scenario_Dashboard is in active use, THE Scenario_Dashboard SHALL render PINN-predicted thermal consequences for a planner override within 5 seconds for any single block change.

---

### Requirement 15: Input Validation and Error Handling

**User Story:** As a system operator, I want all pipeline inputs validated at ingestion and all errors surfaced with actionable diagnostics, so that bad data does not silently corrupt downstream predictions or zoning outputs.

#### Acceptance Criteria

1. WHEN the Ingestion_Pipeline receives a LIDAR point cloud, THE Ingestion_Pipeline SHALL validate that point density is within the range 20–50 pts/m² for at least 90% of the spatial extent, and SHALL log a warning for any sub-region falling below this threshold.
2. WHEN the Ingestion_Pipeline receives satellite thermal imagery, THE Ingestion_Pipeline SHALL validate that atmospheric emissivity correction metadata is present, and IF it is absent, THEN THE Ingestion_Pipeline SHALL reject the image and log a structured error identifying the missing metadata field.
3. WHEN the UCM_Builder receives a UDT block, THE UCM_Builder SHALL validate that all required morphological input fields (building height, footprint area, block area) are non-null and within physically plausible ranges (building height 0–600m, block area > 0 m²).
4. IF the PINN_Model receives an input feature vector containing values outside the training distribution (defined as beyond 3 standard deviations from the training set mean for any feature), THEN THE PINN_Model SHALL return the prediction accompanied by an out-of-distribution warning flag.
5. IF the Optimizer receives constraint parameters that are mutually infeasible (e.g., minimum green cover + minimum FAR exceeding available block area), THEN THE Optimizer SHALL return a structured infeasibility report identifying the conflicting constraints before attempting optimization.
