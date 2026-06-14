# Phase 6R.2 Failure Root Cause Analysis

## Classification: A) Edge Explosion + E) OS OOM Killer

Both causes are confirmed. The edge explosion is the **primary** root cause. The OOM kill is the **mechanism** of termination.

---

## Chain of Causation

```
GeoJSON files use EPSG:4326 (lat/lon degrees)
         ↓
All pairwise building distances are < 0.01 degrees
         ↓
Distance threshold of 30m applied to degree-valued coordinates
         ↓
30.0 >> 0.01 → EVERY building pair satisfies the threshold
         ↓
Graphs become 100% fully connected (N² edges)
         ↓
Archetype 06: 851 nodes → 723,350 edges (per graph)
         ↓
batch_size=32 → one batch contains ~29 graphs → ~12,000,000 edges
         ↓
GATv2Conv computes per-edge attention intermediates
         ↓
Estimated peak memory: ~20 GB (forward + backward + gradients)
         ↓
Available RAM: ~11 GB
         ↓
Linux kernel OOM killer terminates python process (SIGKILL)
         ↓
IDE integrated terminal session destroyed
         ↓
IDE process tree (Chromium-based) also killed by OOM cascading
```

---

## Evidence

### 1. Coordinate System Proof

```
Archetype 06 (851 buildings):
  CRS: EPSG:4326
  X span: 0.005969 degrees (~530 meters)
  Y span: 0.004495 degrees (~500 meters)
  Max pairwise distance: 0.007 degrees
  
  Edges at threshold 30.0: 723,350 (100% of all possible edges)
  Edges at threshold 0.005: 702,030 (97.1% of all possible edges)
```

The 30-meter threshold is expressed in units of meters, but applied to coordinates measured in degrees. Since 1 degree ≈ 111 km, a threshold of 30.0 degrees would encompass the entire planet. The graphs are fully connected.

### 2. Kernel OOM Kill Log (from `/var/log/syslog`)

```
2026-06-14T19:21:13 kernel: oom-kill:constraint=CONSTRAINT_NONE,
  task=python,pid=17137,uid=1000
2026-06-14T19:21:13 kernel: Out of memory: Killed process 17137 (python)
  total-vm:23286076kB, anon-rss:13514856kB
```

The Python process consumed **13.5 GB of anonymous RSS** before being killed, confirming memory exhaustion.

### 3. IDE Crash Explanation

```
2026-06-14T19:21:13 systemd: user@1000.service:
  A process of this unit has been killed by the OOM killer.
2026-06-14T19:21:14 antigravity.desktop:
  [UtilityProcess type: ptyHost, pid: 15646]: unable to kill the process
2026-06-14T19:21:14 antigravity.desktop:
  [UtilityProcess type: shared-process, pid: 15648]:
  crashed with code 15 and reason 'killed'
2026-06-14T19:21:14 antigravity.desktop:
  [UtilityProcess id: 1, type: extensionHost, pid: 15519]:
  crashed with code 15 and reason 'killed'
2026-06-14T19:21:16 systemd:
  app-org.chromium.Chromium-15091.scope: Failed with result 'oom-kill'.
```

**The IDE closes because**: When the Python process exhausts all RAM, the Linux OOM killer selects processes to terminate. It kills Python first (highest RSS). But the memory pressure also triggers kills on the IDE's `ptyHost` (terminal process), `shared-process`, and `extensionHost` subprocesses. Since these are critical IDE subsystems, the entire IDE window closes.

### 4. Model Architecture is Innocent

The EdgeEnhancedGAT has only **~4,419 parameters**. The model weights occupy < 20 KB. The crash has nothing to do with model complexity — it is entirely caused by the edge volume flowing through the attention mechanism.

---

## What Was NOT the Cause

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| B) DataLoader memory duplication | **EXCLUDED** | Default settings (num_workers=0, pin_memory=False). No worker forking. |
| C) GAT attention tensor explosion | **CONTRIBUTING but SECONDARY** | GAT attention amplifies the edge explosion, but the root cause is that edges exist at all. Even a simple message-passing GNN (GraphSAGE) would OOM with 12M edges per batch. |
| D) PyTorch Geometric bug/crash | **EXCLUDED** | No segfault, no CUDA error. Clean SIGKILL from kernel. |
| F) Other | **EXCLUDED** | Kernel logs confirm OOM kill with measured 13.5 GB RSS. |

---

## Required Fix (Do Not Implement Yet — Diagnosis Only)

The graph construction code must **reproject coordinates from EPSG:4326 to a local metric CRS** (e.g., UTM Zone 18N for the Bronx, NY area) before computing pairwise distances. This will convert coordinates from degrees to meters, making the 30m/50m thresholds meaningful.

After reprojection, expected edge densities:
- 30m threshold: ~5–15 edges/node (sparse, physically meaningful)
- 50m threshold: ~10–30 edges/node (moderate)
- Aerodynamic cone: ~3–10 edges/node (optimal for wake physics)

This would reduce batch edge counts from ~12,000,000 to ~50,000–150,000, a **100× reduction**, bringing peak memory to ~0.2 GB — well within system limits.
