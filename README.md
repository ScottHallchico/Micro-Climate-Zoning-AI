# Micro-Climate Zoning AI

A five-phase pipeline that transforms raw urban sensor data into physics-grounded, block-level zoning legislation — treating the city as a living thermodynamic system.

## Architecture

```
Phase 1: Data Ingestion     → Urban Digital Twin (UDT)
Phase 2: Microclimate Model → UCM + Canyon Classifications
Phase 3: PINN Core          → Differentiable Surrogate Model
Phase 4: Optimization       → Pareto-optimal Configurations
Phase 5: Governance         → Zoning Directives + API + Dashboard
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Start Governance API
uvicorn src.governance.api:app --reload

# Start Dashboard (frontend)
cd src/dashboard && npm install && npm run dev
```

## Project Structure

```
src/
├── shared/         # Shared types, data models, configs
├── provenance/     # Central provenance store
├── ingestion/      # Phase 1: Multi-source data ingestion
├── ucm/            # Phase 2: Urban Canopy Model builder
├── cfd/            # Phase 2: CFD training data generation
├── pinn/           # Phase 3: Physics-Informed Neural Network
├── optimizer/      # Phase 4: MOBO optimization engine
├── zoning/         # Phase 5: Zoning code generation
├── governance/     # Phase 5: FastAPI governance API
└── dashboard/      # Phase 5: React/Deck.gl scenario dashboard
```

## Technology Stack

| Category | Tools |
|---|---|
| PINN framework | PyTorch, DeepXDE |
| CFD ground truth | OpenFOAM (RANS k-ε) |
| LIDAR processing | PDAL, Open3D |
| Satellite thermal | Google Earth Engine |
| Urban digital twin | CityGML, 3DCityDB |
| Bayesian optimization | BoTorch |
| Corridor graph analysis | NetworkX |
| Spatial zoning database | PostGIS |
| Governance API | FastAPI |
| Planner dashboard | React, Deck.gl |
