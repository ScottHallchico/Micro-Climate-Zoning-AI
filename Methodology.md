# Micro-Climate Zoning AI — Full Methodology

A five-phase pipeline that transforms raw urban sensor data into physics-grounded, block-level zoning legislation — treating the city as a living thermodynamic system.

---

## Phase 1 — Multi-Source Data Ingestion

Fusing LIDAR geometry, satellite thermal imagery, meteorological streams, and material property libraries into a unified urban data lake.

### Data Streams

| Stream | Source | Specification |
|---|---|---|
| Geometric input | Airborne LIDAR | 20–50 pts/m² point clouds; captures roof overhangs, canopy gaps, street-level geometry |
| Thermal input | ECOSTRESS / Landsat 8–9 thermal bands | 30–70m resolution; corrected for atmospheric emissivity |
| Meteorological | AWS station networks | 500m station density; wind speed/direction, humidity, solar irradiance at 5-min intervals |
| Material properties | ECOSTRESS Spectral Library | Albedo (α), thermal emissivity (ε), volumetric heat capacity (ρCp) per surface type |

### Key Preprocessing Steps

1. LIDAR point cloud ground-classification using Cloth Simulation Filter (CSF), then building polygon extrusion via alpha-shape reconstruction to produce LoD-2 3D meshes.
2. Thermal image co-registration to LIDAR extent using homography transforms; cloud-masking via NDVI × NDWI band combinations.
3. Material classification on roof and pavement surfaces using a Random Forest trained on spectral signatures from the ECOSTRESS Spectral Library (5,000+ materials).
4. Temporal alignment: all streams resampled to a unified 1-hour timestep using kriging interpolation for station-sparse zones.

### Output

The output of Phase 1 is a voxelized **Urban Digital Twin (UDT)** — a 2m × 2m × 2m grid where each cell carries geometry, material properties, and boundary thermal conditions. This is the substrate for all downstream computation.

---

## Phase 2 — 3D Urban Microclimate Modelling

Building the city's thermodynamic baseline using the Urban Canopy Model and street-canyon geometry classification.

### Urban Canopy Model (UCM) Construction

Each city block is parameterized into a UCM using morphological indices derived from the UDT. These indices govern how the block exchanges heat and momentum with the atmosphere above it.

Key indices:
- **Sky View Factor (SVF)** — fraction of visible sky from street level
- **Plan Area Fraction (λp)** — ratio of building footprint to total block area
- **Frontal Area Index (λf)** — building silhouette area per unit ground area
- **Mean Aspect Ratio (H/W)** — canyon height-to-width ratio
- **Roughness length (z₀)** — effective aerodynamic roughness of the block

### Canyon Classification Schema

| Canyon type | H/W ratio | Thermal behavior |
|---|---|---|
| Deep canyon | H/W > 2.5 | Severe re-radiation trapping; single isolated vortex; wind speed reduction >70% |
| Regular canyon | 1 ≤ H/W ≤ 2.5 | Stable counter-rotating vortex pair; primary optimization target for wind corridor design |
| Shallow canyon | H/W < 1 | Skimming flow regime; highest SVF; greatest direct radiation load on streets |
| Transitional zone | Variable | Wake interference between blocks; modelled with stochastic turbulence perturbations |

### Baseline CFD Simulation (Training Data Generation)

Traditional CFD (RANS k-ε turbulence model, OpenFOAM) is run at coarse resolution on 200 stratified samples of the city morphology space. This is expensive — 48–72 hrs per run — but done once to generate the ground-truth dataset for PINN training in Phase 3.

| Metric | Value |
|---|---|
| CFD training samples | 200 |
| Time per CFD run | 48–72 hours |
| PINN inference time (post-training) | ~0.3 seconds |

---

## Phase 3 — Physics-Informed Neural Networks (PINN) Core

The intelligence layer. A neural network that simultaneously learns from CFD training data and respects the Navier-Stokes and energy transport equations as hard constraints.

### Architecture

A multi-head PINN with two prediction branches sharing a common encoder. The encoder takes the block's geometric and material feature vector as input.

- **Shared encoder:** 8-layer MLP, 512 neurons/layer, SIREN activations
- **Wind head:** outputs (u, v, w, P) velocity and pressure field at every voxel
- **Energy head:** outputs (Qh, Qe, ΔQs, Q*) surface energy balance field at every voxel

> **Why SIREN activations?** Standard ReLU networks struggle to represent the smooth, continuous fields required by PDE constraints. SIREN (Sinusoidal Representation Networks) use sin(·) activations, which have well-defined derivatives of all orders — critical for computing ∇²u and ∂T/∂t inside the loss function without numerical instability.

### Physics Constraints Embedded in the Loss Function

```
L_total = L_data + λ₁·L_NS + λ₂·L_energy + λ₃·L_BC

L_NS (Navier-Stokes residuals):
  Continuity:    ∇·u = 0
  Momentum:      ρ(u·∇)u = -∇P + μ∇²u + ρg·β(T - T∞)
  → Buoyancy term β couples wind field to thermal field

L_energy (Urban surface energy balance):
  Q* = Qh + Qe + ΔQs
  Q* = (1-α)K↓ + ε(L↓ - σT⁴)
  → Albedo α, emissivity ε are learned per-material

L_BC (Boundary conditions):
  No-slip at building walls
  Logarithmic wind profile at domain top
  Periodic lateral boundaries for city-scale runs
```

### Transfer Learning Strategy

The base PINN is pre-trained on the 200 CFD samples from Phase 2. For each new city, only the final 2 layers of each head are fine-tuned using 10–15 local weather station measurements — achieving high fidelity from sparse real-world data.

### Multi-Scenario Inference

The trained PINN runs in under 0.3 seconds per block configuration, enabling Monte Carlo sweeps across thousands of urban scenarios — comparing current vs. proposed zoning configurations in real time.

Scenarios modelled:
- Diurnal cycle (24 hrs, 15-min steps)
- Seasonal variation (4 canonical days)
- Extreme heat events (90th-percentile temperature days)
- Future climate projections (RCP 4.5 / 8.5)

---

## Phase 4 — Thermodynamic Optimization Engine

Uses the PINN as a differentiable surrogate to search the urban morphology space for configurations that minimize UHI intensity, subject to real-world constraints.

### Objective Function

```
Minimize: UHI_intensity(x)
  = Σ_blocks [ T_block(x) - T_rural ] · Pop_density_weight

Subject to:
  x ∈ X_feasible               (physical morphology constraints)
  FAR_block ≤ FAR_max          (Floor Area Ratio limit)
  ΔH_block ≤ ΔH_permitted      (max height change from baseline)
  Wind_corridor[k] ≥ V_min     (for all k ∈ identified corridors)
  Green_cover ≥ 0.15 × block_area   (minimum 15% greenery)
  Σ_blocks cost(x) ≤ budget_total
```

### Optimization Algorithm

Because the PINN is fully differentiable (via autograd), the optimizer uses gradient information directly — treating the network's Jacobian as a free sensitivity analysis of the city's thermal response to any morphological change.

| Component | Detail |
|---|---|
| Algorithm | Multi-Objective Bayesian Optimization (MOBO) with PINN-computed gradients as acquisition function priors |
| Decision variables (x) | Building heights, setbacks, roof albedo targets, green-roof fraction, street tree canopy %, pavement material |
| Secondary objectives | Maximize pedestrian thermal comfort (PET index); preserve solar access ≥4 h/day; minimize retrofit cost per block |
| Output | Pareto front of 50–200 non-dominated urban configurations; planners choose from this frontier |

### Wind Corridor Preservation Logic

The optimizer explicitly identifies and protects dominant wind vectors at the city scale using a graph-based corridor extraction algorithm. Street segments are nodes; airflow connectivity is the edge weight. A minimum-cut analysis finds the critical bottleneck blocks — these receive hard height caps that cannot be overridden by other objectives.

> **Critical design constraint:** Wind corridors are directional and pressure-driven. A building height increase of only 2 floors in a transitional zone can cause the Bernoulli effect to redirect flow by 35–50°, effectively invalidating the cooling benefit of blocks 400m downwind. The optimizer propagates this sensitivity via PINN Jacobians before any height increase is proposed.

---

## Phase 5 — Zoning Code Generation & Governance Interface

Translates the optimizer's Pareto-optimal configurations into machine-readable, legally structured zoning amendments — with full justification trails for democratic review.

### Sample Generated Zoning Directives

**Block 4 — Corridor Preservation Zone** *(Priority: Critical)*

```json
{
  "block_id": "BLK-04",
  "zone_class": "WIND_CORRIDOR_CRITICAL",
  "max_height_m": 12,
  "height_justification": "Eastern wind corridor (vector 087°, avg 3.2 m/s) would be
    deflected by >40° if height exceeds 12m. Downstream cooling loss: +1.8°C on
    6 dependent blocks.",
  "permitted_uses": ["residential_low", "retail_ground_floor"],
  "mandatory_interventions": [],
  "review_trigger": "Annual wind-corridor re-assessment via PINN update"
}
```

*Rationale: This block sits at the inlet of the primary summer wind corridor. Height restriction is the minimum intervention needed to preserve downstream cooling for blocks 5–11.*

---

**Block 9 — Heat Accumulation Hotspot** *(Priority: High)*

```json
{
  "block_id": "BLK-09",
  "zone_class": "THERMAL_REMEDIATION",
  "max_height_m": null,
  "mandatory_interventions": [
    {
      "type": "green_roof",
      "min_coverage_pct": 60,
      "species": "sedum_mosaic",
      "cooling_effect_estimate": "-0.9°C block surface temp"
    },
    {
      "type": "pavement_albedo",
      "min_albedo": 0.5,
      "current_albedo": 0.12,
      "retrofit_priority": "streets_first"
    },
    {
      "type": "street_tree_canopy",
      "min_coverage_pct": 35,
      "placement": "south_and_west_facades"
    }
  ],
  "compliance_deadline_months": 36,
  "penalty_per_non_compliant_sqm": "₹850/year"
}
```

*Rationale: Block 9 has an SVF of 0.81 (very open sky), trapping direct solar radiation due to its concave street geometry. No building height change needed — surface material interventions achieve the equivalent cooling effect at 40% lower social disruption cost.*

---

**Block 17 — Adaptive Density Opportunity** *(Priority: Moderate)*

```json
{
  "block_id": "BLK-17",
  "zone_class": "DENSITY_ADAPTIVE",
  "current_max_height_m": 24,
  "proposed_max_height_m": 36,
  "condition": "Taller massing on this block increases canyon shading and reduces
    street-level Tmrt by 2.1°C with no corridor impact. FAR increase conditional on:
    min_albedo_roof=0.65, green_podium_required=true",
  "thermal_benefit": "+2.1°C pedestrian comfort improvement",
  "economic_co_benefit": "18% additional FAR unlocked"
}
```

*Rationale: Counter-intuitive finding — increasing height here generates beneficial self-shading. The block's orientation creates a cool corridor effect. The PINN identified this non-obvious opportunity that conventional zoning would have blocked.*

---

### Governance & Democratic Interface

| Layer | Mechanism | Who interacts |
|---|---|---|
| Justification trail | Every directive links to its PINN prediction, CFD training sample, and optimization run — full auditability | Urban planners, courts |
| Scenario dashboard | Interactive map where planners can override any directive and see the PINN-predicted thermal consequence on all affected blocks in real time | City planning dept. |
| Community heat index | Per-ward aggregate of cooling benefit, cost burden, and displacement risk — flags equity concerns automatically | Elected officials, public |
| Rolling update cycle | PINN re-trained quarterly with new AWS + satellite data; zoning codes auto-flagged for review if thermal model drift > 0.4°C | AI operations team |
| Developer compliance API | REST endpoint; building permit applications checked against current zoning codes in real time before submission | Architects, developers |

> **Core design principle:** The AI does not make zoning decisions — it generates evidence-grounded options with quantified trade-offs. Human planners and elected bodies retain the legal authority to adopt, modify, or reject any directive. The system is designed to make the physics visible, not to replace democratic governance.

---

## Technology Stack

| Category | Tools |
|---|---|
| PINN framework | PyTorch, DeepXDE |
| CFD ground truth | OpenFOAM (RANS k-ε) |
| LIDAR processing | PDAL, Open3D |
| Satellite thermal | Google Earth Engine, NASA LP DAAC |
| Urban digital twin | CityGML, 3DCityDB |
| Bayesian optimization | BoTorch |
| Corridor graph analysis | NetworkX |
| Spatial zoning database | PostGIS |
| Governance API | FastAPI |
| Planner dashboard | React, Deck.gl |