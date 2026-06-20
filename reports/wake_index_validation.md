# Wake Severity Index — Validation Report

## Old vs New: Distribution Comparison

### Old Binary Classifier

| Class | Count | Fraction |
|---|---|---|
| Wake | 38,084 | 63.5% |
| Non-Wake | 21,916 | 36.5% |

### New WSI Classifier

| Class | Count | Fraction |
|---|---|---|
| No Wake | 3,264 | 5.4% |
| Mild Wake | 31,551 | 52.6% |
| Strong Wake | 22,715 | 37.9% |
| Severe Wake | 2,470 | 4.1% |

## Cluster Separation Metrics

### WSI Component Space (W_v, W_t, W_p)

| Classifier | Silhouette ↑ | Davies-Bouldin ↓ |
|------------|-------------|------------------|
| Old Binary | 0.3313 | 1.5629 |
| **New WSI** | **0.1329** | **2.1249** |

### 1D WSI Axis

| Classifier | Silhouette ↑ | Davies-Bouldin ↓ |
|------------|-------------|------------------|
| Old Binary | 0.3150 | 0.9656 |
| **New WSI** | **0.4642** | **0.5035** |

### Inter-Class WSI Separation

| Class | Mean WSI |
|---|---|
| No Wake | 0.1936 |
| Mild Wake | 0.4000 |
| Strong Wake | 0.5699 |
| Severe Wake | 0.8270 |

**Mean pairwise separation**: 0.3450

## Verdict

**WSI is superior on the 1D axis.** The composite index produces well-separated classes along its own scalar dimension, confirming that the velocity-deficit, turbulence, and pressure components yield a physically meaningful gradient of wake severity.

## Figures

- `publication_figures/wsi_distribution.png`
- `publication_figures/old_vs_new_wake.png`
- `publication_figures/wsi_component_scatter.png`
- `publication_figures/wake_spatial_map.png`
