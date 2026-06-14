# Smart CFD Execution Schedule

## Tiers & Prioritization

**Tier 1 (Fast)**: $U \leq 6$ m/s, Density < 0.4 (Archetypes 03, 10)
- Characteristics: Simple wakes, fast convergence (~20 mins).
- Count: ~192 cases.
- Priority: Highest (Baseline mapping).

**Tier 2 (Moderate)**: $U > 6$ m/s, Density < 0.4 OR $U \leq 6$ m/s, Density $\geq$ 0.4
- Characteristics: Moderate flow separation, stable wakes (~40 mins).
- Count: ~384 cases.
- Priority: Medium (Fills interpolation space).

**Tier 3 (Expensive)**: $U > 6$ m/s, Density $\geq$ 0.4 (Archetypes 06, 09)
- Characteristics: Intense canyon effects, high turbulence, slow convergence (~60-90 mins).
- Count: ~192 cases.
- Priority: Lowest (Run overnight/weekends).

## Resource Estimation
- Average Core-Hours per case: 1.0 hr (assuming 8 cores).
- Total CPU Time: ~768 compute hours.
- Peak Memory: 4-12 GB depending on Archetype.

## Checkpointing Strategy
- Results saved to Parquet after EVERY simulation.
- `simpleFoam` state writes disabled to save disk space, only final fields kept.
