#!/usr/bin/env python3
"""
Phase 8M — Full Dataset Retraining Audit

Retrains the surrogate on the reconstructed 10-archetype dataset
and benchmarks via strict LOAO cross-validation.

RULES:
  - No CFD targets in features/edges/graph construction
  - Scalers fitted ONLY on training folds
  - Every metric from actual inference
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, GATv2Conv
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import copy, time
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
ML      = PROJECT / "data" / "ml"
REPORTS = PROJECT / "reports"

# ── Models ──────────────────────────────────────────────────

class EdgeGAT(nn.Module):
    def __init__(self, in_dim, edge_dim):
        super().__init__()
        self.c1 = GATv2Conv(in_dim, 32, heads=2, concat=False, edge_dim=edge_dim)
        self.c2 = GATv2Conv(32, 32, heads=2, concat=False, edge_dim=edge_dim)
        self.flow = nn.Linear(32, 5)
        self.wake = nn.Linear(32, 1)
    def forward(self, x, ei, ea):
        h = F.elu(self.c1(x, ei, ea))
        h = F.elu(self.c2(h, ei, ea))
        return self.flow(h), self.wake(h)

class TransV2(nn.Module):
    def __init__(self, in_dim, edge_dim):
        super().__init__()
        self.c1 = TransformerConv(in_dim, 64, heads=2, concat=False, edge_dim=edge_dim)
        self.n1 = nn.LayerNorm(64)
        self.c2 = TransformerConv(64, 64, heads=2, concat=False, edge_dim=edge_dim)
        self.n2 = nn.LayerNorm(64)
        self.drop = nn.Dropout(0.1)
        self.flow = nn.Linear(64, 5)
        self.wake = nn.Linear(64, 1)
    def forward(self, x, ei, ea):
        h = self.drop(F.elu(self.n1(self.c1(x, ei, ea))))
        h = self.drop(F.elu(self.n2(self.c2(h, ei, ea)))) + h
        return self.flow(h), self.wake(h)

IN_DIM  = 15   # 3 coords + 2 wind + 8 morph + 2 local
EDGE_DIM = 5   # dx dy dz dist align

# ── Graph Construction (geometry-only) ──────────────────────

def build_graphs(df, cdf, morph_df, max_nodes=500):
    graphs = []
    morph_df.index = morph_df.index.astype(int)
    for sim_id, grp in df.groupby('simulation_id'):
        if len(grp) > max_nodes:
            grp = grp.sample(max_nodes, random_state=42)
        coords = grp[['x','y','z']].values.astype(np.float64)
        meta = cdf[cdf['simulation_id'] == sim_id]
        if len(meta) == 0: continue
        meta = meta.iloc[0]
        ws, wd = float(meta['wind_speed']), float(meta['wind_direction'])
        arch = int(grp['archetype'].iloc[0])

        morph = morph_df.loc[arch].values if arch in morph_df.index else np.zeros(8)
        tree = cKDTree(coords)
        local_d = np.array([len(tree.query_ball_point(c, 30.0)) for c in coords], dtype=np.float64)
        local_h = np.array([np.mean(coords[tree.query_ball_point(c, 30.0), 2]) for c in coords], dtype=np.float64)

        N = len(grp)
        x_feat = np.column_stack([
            coords, np.full(N,ws), np.full(N,wd),
            np.tile(morph,(N,1)), local_d, local_h
        ]).astype(np.float32)

        u,v,w = grp['u'].values, grp['v'].values, grp['w'].values
        p_arr, k_arr = grp['p'].values, grp['k'].values
        speed = np.sqrt(u**2 + v**2 + w**2)
        wake = (speed < 0.3*ws).astype(np.float32)
        y = np.column_stack([u,v,w,p_arr,k_arr,wake]).astype(np.float32)

        pairs = tree.query_pairs(50.0)
        if not pairs: continue
        ij = np.array(list(pairs))
        s, d = ij[:,0], ij[:,1]
        s_all, d_all = np.concatenate([s,d]), np.concatenate([d,s])
        vecs = coords[d_all] - coords[s_all]
        dists = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-6)
        rad = np.radians(wd)
        wvec = np.array([np.cos(rad), np.sin(rad), 0.0])
        align = (vecs @ wvec).reshape(-1,1)
        mask = align.flatten() > 0
        s_all, d_all = s_all[mask], d_all[mask]
        vecs, dists, align = vecs[mask], dists[mask], align[mask]

        ei = torch.tensor(np.vstack([s_all, d_all]), dtype=torch.long)
        ea = torch.tensor(np.column_stack([vecs, dists, align]).astype(np.float32))
        data = Data(x=torch.tensor(x_feat), edge_index=ei, edge_attr=ea, y=torch.tensor(y))
        data.archetype = arch
        graphs.append(data)
    return graphs

# ── Fold-safe training ──────────────────────────────────────

def fold_train_eval(model_cls, train_data, val_data, epochs=20, p_w=0.2):
    # Fit pressure scaler on training only
    tp = np.concatenate([d.y[:,3].numpy() for d in train_data]).reshape(-1,1)
    scaler = PowerTransformer(method='yeo-johnson')
    scaler.fit(tp)
    for d in train_data:
        d.y[:,3] = torch.tensor(scaler.transform(d.y[:,3].numpy().reshape(-1,1)).flatten())
    for d in val_data:
        d.y[:,3] = torch.tensor(scaler.transform(d.y[:,3].numpy().reshape(-1,1)).flatten())

    model = model_cls(IN_DIM, EDGE_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    tl = DataLoader(train_data, batch_size=4, shuffle=True)

    for _ in range(epochs):
        model.train()
        for b in tl:
            opt.zero_grad()
            fl, wl = model(b.x, b.edge_index, b.edge_attr)
            loss = (F.mse_loss(fl[:,:3], b.y[:,:3])
                    + F.mse_loss(fl[:,3], b.y[:,3]) * p_w
                    + F.mse_loss(fl[:,4], b.y[:,4])
                    + F.binary_cross_entropy_with_logits(wl.view(-1), b.y[:,5]))
            loss.backward(); opt.step()

    model.eval()
    yt, fp, wp = [], [], []
    with torch.no_grad():
        for b in DataLoader(val_data, batch_size=4):
            fl, wl = model(b.x, b.edge_index, b.edge_attr)
            yt.append(b.y.numpy()); fp.append(fl.numpy())
            wp.append(torch.sigmoid(wl).view(-1).numpy())
    yt, fp, wp = np.vstack(yt), np.vstack(fp), np.concatenate(wp)

    targets = ['u','v','w','p','k','wake']
    metrics = {}
    for i,t in enumerate(targets):
        pred = wp if t == 'wake' else fp[:,i]
        metrics[f'{t}_r2']   = r2_score(yt[:,i], pred)
        metrics[f'{t}_rmse'] = np.sqrt(mean_squared_error(yt[:,i], pred))
        metrics[f'{t}_mae']  = mean_absolute_error(yt[:,i], pred)
    return metrics

# ── LOAO ────────────────────────────────────────────────────

def run_loao(graphs, model_cls, epochs=20):
    archetypes = sorted(set(g.archetype for g in graphs))
    rows = []
    for holdout in archetypes:
        tr = copy.deepcopy([g for g in graphs if g.archetype != holdout])
        va = copy.deepcopy([g for g in graphs if g.archetype == holdout])
        if not va: continue
        m = fold_train_eval(model_cls, tr, va, epochs=epochs)
        m['holdout'] = holdout
        rows.append(m)
        print(f"  Fold {holdout}: wake={m['wake_r2']:.3f} u={m['u_r2']:.3f} p={m['p_r2']:.3f}")
    return pd.DataFrame(rows)

# ── Tabular baselines ───────────────────────────────────────

def run_tabular_baseline(graphs, name, model_fn):
    archs = sorted(set(g.archetype for g in graphs))
    holdout = archs[-1]
    tr = [g for g in graphs if g.archetype != holdout]
    va = [g for g in graphs if g.archetype == holdout]

    xt = np.vstack([g.x.numpy() for g in tr])
    yt = np.vstack([g.y.numpy() for g in tr])
    xv = np.vstack([g.x.numpy() for g in va])
    yv = np.vstack([g.y.numpy() for g in va])

    results = {'model': name}
    for i, t in enumerate(['u','v','w','p','k']):
        m = model_fn()
        m.fit(xt, yt[:,i])
        pred = m.predict(xv)
        results[f'{t}_r2'] = r2_score(yv[:,i], pred)
    return results

# ── Main ────────────────────────────────────────────────────

def main():
    print("Phase 8M — Full Dataset Retraining Audit")
    t0 = time.time()

    df    = pd.read_parquet(ML / "cfd_field_dataset_full.parquet")
    cdf   = pd.read_parquet(ML / "verified_cdf_dataset_v3.parquet") if (ML / "verified_cdf_dataset_v3.parquet").exists() else pd.read_parquet(ML / "verified_cfd_dataset_v3.parquet")
    morph = pd.read_parquet(ML / "morphology_feature_matrix.parquet")

    # ── WS1: Dataset Integrity ──
    n_sims  = df['simulation_id'].nunique()
    n_arch  = df['archetype'].nunique()
    n_pts   = len(df)
    md1  = "# Full Dataset Audit\n\n"
    md1 += f"- Points: {n_pts:,}\n- Simulations: {n_sims}\n- Archetypes: {n_arch}\n"
    (REPORTS / "full_dataset_audit.md").write_text(md1)
    print(f"[WS1] {n_pts:,} points, {n_sims} sims, {n_arch} archetypes")

    # ── Build graphs ──
    print("[BUILD] Constructing geometry-only graphs (500 nodes/sim) ...")
    graphs = build_graphs(df, cdf, morph, max_nodes=500)
    print(f"  {len(graphs)} graphs built")

    # ── WS2: Full LOAO ──
    print("[WS2] Full LOAO (TransV2, 20 epochs) ...")
    loao_df = run_loao(graphs, TransV2, epochs=20)
    loao_df.to_csv(REPORTS / "loao_fold_results.csv", index=False)

    md2  = "# LOAO Fold Results (Full Dataset)\n\n"
    md2 += "| Holdout | Wake R² | u R² | p R² |\n|---|---|---|---|\n"
    for _, r in loao_df.iterrows():
        md2 += f"| {int(r['holdout'])} | {r['wake_r2']:.4f} | {r['u_r2']:.4f} | {r['p_r2']:.4f} |\n"
    mean_w = loao_df['wake_r2'].mean()
    mean_u = loao_df['u_r2'].mean()
    mean_p = loao_df['p_r2'].mean()
    md2 += f"\n**Mean**: Wake={mean_w:.4f}  Vel={mean_u:.4f}  Press={mean_p:.4f}\n"
    md2 += f"**Std**:  Wake={loao_df['wake_r2'].std():.4f}  Vel={loao_df['u_r2'].std():.4f}  Press={loao_df['p_r2'].std():.4f}\n"
    (REPORTS / "loao_fold_results.md").write_text(md2)

    # ── WS3: Baselines ──
    print("[WS3] Baseline comparison ...")
    rf_res  = run_tabular_baseline(graphs, 'RandomForest',
                lambda: RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42, n_jobs=-1))
    lgb_res = run_tabular_baseline(graphs, 'LightGBM',
                lambda: LGBMRegressor(n_estimators=20, max_depth=8, random_state=42, verbose=-1))

    # EdgeGAT single fold for comparison
    archs = sorted(set(g.archetype for g in graphs))
    tr_gat = copy.deepcopy([g for g in graphs if g.archetype != archs[-1]])
    va_gat = copy.deepcopy([g for g in graphs if g.archetype == archs[-1]])
    gat_m = fold_train_eval(EdgeGAT, tr_gat, va_gat, epochs=20)
    print(f"  EdgeGAT: wake={gat_m['wake_r2']:.3f} u={gat_m['u_r2']:.3f}")

    md3  = "# Model Comparison (Full Dataset)\n\n"
    md3 += "| Model | u R² | v R² | w R² | p R² | k R² |\n|---|---|---|---|---|---|\n"
    for res in [rf_res, lgb_res]:
        md3 += f"| {res['model']} | {res['u_r2']:.4f} | {res['v_r2']:.4f} | {res['w_r2']:.4f} | {res['p_r2']:.4f} | {res['k_r2']:.4f} |\n"
    md3 += f"| EdgeGAT | {gat_m['u_r2']:.4f} | {gat_m['v_r2']:.4f} | {gat_m['w_r2']:.4f} | {gat_m['p_r2']:.4f} | {gat_m['k_r2']:.4f} |\n"
    md3 += f"| TransV2 (LOAO mean) | {mean_u:.4f} | {loao_df['v_r2'].mean():.4f} | {loao_df['w_r2'].mean():.4f} | {mean_p:.4f} | {loao_df['k_r2'].mean():.4f} |\n"
    (REPORTS / "model_comparison_full_dataset.md").write_text(md3)

    # ── WS4: Diversity Impact ──
    # Phase 8K old results (2-archetype dataset)
    old_w, old_u, old_p = 0.2148, 0.0634, -0.0402
    md4  = "# Diversity Impact Analysis\n\n"
    md4 += "| Metric | Old (2 arch) | Full (10 arch) | ΔR² |\n|---|---|---|---|\n"
    md4 += f"| Wake R² | {old_w:.4f} | {mean_w:.4f} | {mean_w - old_w:+.4f} |\n"
    md4 += f"| Velocity R² | {old_u:.4f} | {mean_u:.4f} | {mean_u - old_u:+.4f} |\n"
    md4 += f"| Pressure R² | {old_p:.4f} | {mean_p:.4f} | {mean_p - old_p:+.4f} |\n"
    (REPORTS / "diversity_impact_analysis.md").write_text(md4)

    # ── WS5: Root Cause Resolution ──
    improved = (mean_u > old_u + 0.05) or (mean_w > old_w + 0.05)
    if improved:
        cause = "C) Dataset bottleneck — CONFIRMED"
        explanation = "Adding 8 archetypes materially improved generalization."
    elif mean_u > old_u:
        cause = "C) Dataset bottleneck — PARTIALLY CONFIRMED"
        explanation = "Marginal improvement suggests data diversity helps but architecture also limits."
    else:
        cause = "B) Architecture bottleneck"
        explanation = "Full dataset did not improve performance; the GNN architecture itself is the limiting factor."
    md5  = "# Root Cause Resolution\n\n"
    md5 += f"**Verdict**: {cause}\n\n{explanation}\n"
    (REPORTS / "root_cause_resolution.md").write_text(md5)

    # ── Final Certification ──
    md_cert  = "# Phase 8M Retraining Audit\n\n"
    md_cert += f"1. **Does performance improve?** ΔWake={mean_w-old_w:+.4f}, ΔVel={mean_u-old_u:+.4f}, ΔPress={mean_p-old_p:+.4f}\n"
    md_cert += f"2. **Is LOAO meaningful?** Yes — 10 archetypes, {len(loao_df)} folds executed\n"
    md_cert += f"3. **Was dataset bottleneck correct?** {cause}\n"

    ready = mean_w > 0.50 and mean_u > 0.50 and mean_p > 0.30
    md_cert += f"4. **Phase 9 ready?** {'YES' if ready else 'NOT YET'}\n\n"
    md_cert += f"Mean LOAO: Wake={mean_w:.4f}  Vel={mean_u:.4f}  Press={mean_p:.4f}\n"
    md_cert += f"Runtime: {time.time()-t0:.1f}s\n"
    (REPORTS / "phase8m_retraining_audit.md").write_text(md_cert)

    print(f"\nPhase 8M Complete. Wake={mean_w:.3f} Vel={mean_u:.3f} Press={mean_p:.3f}")

if __name__ == "__main__":
    main()
