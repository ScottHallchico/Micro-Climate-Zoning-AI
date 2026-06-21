#!/usr/bin/env python3
"""
Phase 8P — Representation Replacement Study
"""

import numpy as np
import pandas as pd
import time
import copy
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, GATv2Conv, global_max_pool, MessagePassing
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
ML      = PROJECT / "data" / "ml"
REPORTS = PROJECT / "reports"

# ── Models ─────────────────────────────────────────────────────────

class SimplePointConv(MessagePassing):
    def __init__(self, nn):
        super().__init__(aggr='max')
        self.nn = nn
    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)
    def message(self, x_j, pos_i, pos_j):
        pos_diff = pos_j - pos_i
        msg = torch.cat([x_j, pos_diff], dim=-1)
        return self.nn(msg)

class PointNetPP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp_local = nn.Sequential(
            nn.Linear(in_dim + 3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.conv = SimplePointConv(self.mlp_local)
        self.mlp_global = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.head_flow = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 4) # u, v, w, Cp
        )
        self.head_wake = nn.Sequential(
            nn.Linear(256 + 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, pos, edge_index, edge_attr, batch):
        h_local = self.conv(x, pos, edge_index)
        h_global = self.mlp_global(h_local)
        g = global_max_pool(h_global, batch)
        g_expand = g[batch]
        out = torch.cat([h_local, g_expand], dim=1)
        return self.head_flow(out), self.head_wake(out)

class EdgeGAT(nn.Module):
    def __init__(self, in_dim, edge_dim):
        super().__init__()
        self.c1 = GATv2Conv(in_dim, 32, heads=2, concat=False, edge_dim=edge_dim)
        self.c2 = GATv2Conv(32, 32, heads=2, concat=False, edge_dim=edge_dim)
        self.flow = nn.Linear(32, 4)
        self.wake = nn.Linear(32, 1)
    def forward(self, x, pos, ei, ea, batch):
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
        self.flow = nn.Linear(64, 4)
        self.wake = nn.Linear(64, 1)
    def forward(self, x, pos, ei, ea, batch):
        h = F.elu(self.n1(self.c1(x, ei, ea)))
        h = F.elu(self.n2(self.c2(h, ei, ea))) + h
        return self.flow(h), self.wake(h)


# ── Data Prep ───────────────────────────────────────────────────────

def prepare_data(df, cdf, morph_df, max_nodes=500):
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
        
        # Relative Geometry
        x_min, x_max = coords[:,0].min(), coords[:,0].max()
        y_min, y_max = coords[:,1].min(), coords[:,1].max()
        z_max = coords[:,2].max()
        
        x_norm = (coords[:,0] - x_min) / (x_max - x_min + 1e-6)
        y_norm = (coords[:,1] - y_min) / (y_max - y_min + 1e-6)
        z_norm = coords[:,2] / (z_max + 1e-6)
        
        tree = cKDTree(coords)
        local_d = np.array([len(tree.query_ball_point(c, 50.0)) for c in coords], dtype=np.float64)
        local_h_list = [coords[tree.query_ball_point(c, 50.0), 2] for c in coords]
        local_hm = np.array([np.mean(h) if len(h)>0 else c[2] for c, h in zip(coords, local_h_list)])
        local_hv = np.array([np.var(h) if len(h)>0 else 0.0 for h in local_h_list])
        
        z_max_idx = np.argmax(coords[:, 2])
        tallest_pt = coords[z_max_idx]
        dist_tallest = np.linalg.norm(coords[:, :2] - tallest_pt[:2], axis=1)
        
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        dist_center = np.linalg.norm(coords[:, :2] - np.array([x_center, y_center]), axis=1)
        
        rad = np.radians(wd)
        wvec = np.array([np.cos(rad), np.sin(rad)])
        
        N = len(grp)
        # x_feat: [x,y,z, ws, wd, morph(8), local_d, local_hm, local_hv, dist_tallest, x_norm, y_norm, z_norm, dist_center]
        # Total = 3 + 2 + 8 + 4 + 3 + 1 = 21 features
        x_feat = np.column_stack([
            coords, np.full(N,ws), np.full(N,wd), np.tile(morph,(N,1)),
            local_d, local_hm, local_hv, dist_tallest,
            x_norm, y_norm, z_norm, dist_center
        ]).astype(np.float32)

        u,v,w = grp['u'].values, grp['v'].values, grp['w'].values
        p_arr = grp['p'].values
        speed = np.sqrt(u**2 + v**2 + w**2)
        wake = (speed < 0.3*ws).astype(np.float32)
        p_coef = p_arr / (0.5 * 1.225 * ws**2 + 1e-3)
        y = np.column_stack([u, v, w, p_coef, wake]).astype(np.float32)

        pairs = tree.query_pairs(50.0)
        if not pairs: continue
        ij = np.array(list(pairs))
        s_all, d_all = np.concatenate([ij[:,0],ij[:,1]]), np.concatenate([ij[:,1],ij[:,0]])
        vecs = coords[d_all] - coords[s_all]
        dists = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-6)
        align = (vecs[:,:2] @ wvec).reshape(-1,1)
        
        ei = torch.tensor(np.vstack([s_all, d_all]), dtype=torch.long)
        ea = torch.tensor(np.column_stack([vecs, dists, align]).astype(np.float32))
        
        data = Data(x=torch.tensor(x_feat), pos=torch.tensor(coords).float(), edge_index=ei, edge_attr=ea, y=torch.tensor(y))
        data.archetype = arch
        graphs.append(data)
    return graphs

# ── Train loops ─────────────────────────────────────────────────────

def train_tabular(model_cls, x_tr, y_tr, x_te, y_te):
    res = {}
    for i, t in enumerate(['u', 'wake', 'cp']):
        idx = 0 if t=='u' else (4 if t=='wake' else 3)
        m = model_cls()
        m.fit(x_tr, y_tr[:, idx])
        pred = m.predict(x_te)
        res[f'{t}_r2'] = r2_score(y_te[:, idx], pred)
        res[f'{t}_mae'] = mean_absolute_error(y_te[:, idx], pred)
    return res

def train_nn(model_cls, train_data, val_data, epochs=15):
    # Scale inputs
    tx = np.concatenate([d.x.numpy() for d in train_data])
    x_scaler = StandardScaler().fit(tx)
    for d in train_data: d.x = torch.tensor(x_scaler.transform(d.x.numpy()).astype(np.float32))
    for d in val_data: d.x = torch.tensor(x_scaler.transform(d.x.numpy()).astype(np.float32))

    in_dim = train_data[0].x.shape[1]
    if model_cls == PointNetPP: model = model_cls(in_dim)
    else: model = model_cls(in_dim, 5)

    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    tl = DataLoader(train_data, batch_size=4, shuffle=True)
    vl = DataLoader(val_data, batch_size=4)

    for _ in range(epochs):
        model.train()
        for b in tl:
            opt.zero_grad()
            fl, wl = model(b.x, b.pos, b.edge_index, b.edge_attr, b.batch)
            loss = F.mse_loss(fl[:,:3], b.y[:,:3]) + F.mse_loss(fl[:,3], b.y[:,3]) + F.binary_cross_entropy_with_logits(wl.view(-1), b.y[:,4])
            loss.backward(); opt.step()

    model.eval()
    yt, fp, wp = [], [], []
    with torch.no_grad():
        for b in vl:
            fl, wl = model(b.x, b.pos, b.edge_index, b.edge_attr, b.batch)
            yt.append(b.y.numpy()); fp.append(fl.numpy())
            wp.append(torch.sigmoid(wl).view(-1).numpy())
    yt, fp, wp = np.vstack(yt), np.vstack(fp), np.concatenate(wp)

    res = {}
    res['u_r2'] = r2_score(yt[:,0], fp[:,0])
    res['u_mae'] = mean_absolute_error(yt[:,0], fp[:,0])
    res['cp_r2'] = r2_score(yt[:,3], fp[:,3])
    res['cp_mae'] = mean_absolute_error(yt[:,3], fp[:,3])
    res['wake_r2'] = r2_score(yt[:,4], wp)
    res['wake_mae'] = mean_absolute_error(yt[:,4], wp)
    
    if model_cls == PointNetPP:
        torch.save(model.state_dict(), ML / "pointnetpp_surrogate.pt")
        
    return res, fp, wp, yt

# ── Main ────────────────────────────────────────────────────────────

def main():
    print("Phase 8P — Representation Replacement Study")
    df    = pd.read_parquet(ML / "cfd_field_dataset_full.parquet")
    cdf   = pd.read_parquet(ML / "verified_cfd_dataset_v3.parquet")
    morph = pd.read_parquet(ML / "morphology_feature_matrix.parquet")

    print("[PREP] Building graphs & features...")
    graphs = prepare_data(df, cdf, morph, max_nodes=500)
    archs = sorted(list(set(g.archetype for g in graphs)))

    # Extracted feature indices
    # 0,1,2: x,y,z
    # 3,4: ws, wd
    # 5-12: morph
    # 13,14,15,16: local_d, local_hm, local_hv, dist_tallest
    # 17,18,19: x_norm, y_norm, z_norm
    # 20: dist_center

    X_all = np.vstack([g.x.numpy() for g in graphs])
    Y_all = np.vstack([g.y.numpy() for g in graphs])
    A_all = np.concatenate([np.full(len(g.x), g.archetype) for g in graphs])

    # WS1: Coordinate Leakage Audit
    print("[WS1] Coordinate Leakage Audit (LightGBM)...")
    res_base, res_nocoord = [], []
    for h in archs:
        tr, te = A_all != h, A_all == h
        # Baseline: x,y,z + ws,wd + morph
        idx_base = list(range(13))
        # No-coord: ws,wd + morph
        idx_nocoord = list(range(3,13))
        
        rb = train_tabular(lambda: LGBMRegressor(n_estimators=30, random_state=42, verbose=-1), X_all[tr][:, idx_base], Y_all[tr], X_all[te][:, idx_base], Y_all[te])
        rnc = train_tabular(lambda: LGBMRegressor(n_estimators=30, random_state=42, verbose=-1), X_all[tr][:, idx_nocoord], Y_all[tr], X_all[te][:, idx_nocoord], Y_all[te])
        res_base.append(rb); res_nocoord.append(rnc)

    mb_u = np.mean([r['u_r2'] for r in res_base])
    mn_u = np.mean([r['u_r2'] for r in res_nocoord])
    
    md1 = "# Coordinate Leakage Audit\n\n| Model | Vel R² | Wake R² | Cp R² |\n|---|---|---|---|\n"
    md1 += f"| Baseline (with x,y,z) | {mb_u:.4f} | {np.mean([r['wake_r2'] for r in res_base]):.4f} | {np.mean([r['cp_r2'] for r in res_base]):.4f} |\n"
    md1 += f"| Leakage-Free (no x,y,z) | {mn_u:.4f} | {np.mean([r['wake_r2'] for r in res_nocoord]):.4f} | {np.mean([r['cp_r2'] for r in res_nocoord]):.4f} |\n"
    md1 += f"\n**Absolute Drop**: {mb_u - mn_u:.4f}\n"
    md1 += f"**MEMORIZATION CONFIRMED**" if (mb_u - mn_u) > 0.05 else "**No significant memorization.**"
    (REPORTS / "coordinate_leakage_audit.md").write_text(md1)

    # WS2: Relative Geometry Encoding
    print("[WS2] Relative Geometry Encoding (LightGBM)...")
    res_rel = []
    idx_rel = list(range(3,21)) # skip 0,1,2 (absolute x,y,z), include everything else
    for h in archs:
        tr, te = A_all != h, A_all == h
        rr = train_tabular(lambda: LGBMRegressor(n_estimators=40, random_state=42, verbose=-1), X_all[tr][:, idx_rel], Y_all[tr], X_all[te][:, idx_rel], Y_all[te])
        res_rel.append(rr)
        
    mrel_u = np.mean([r['u_r2'] for r in res_rel])
    md2 = "# Relative Geometry Study\n\n| Target | R² |\n|---|---|\n"
    md2 += f"| Velocity | {mrel_u:.4f} |\n"
    md2 += f"| Wake | {np.mean([r['wake_r2'] for r in res_rel]):.4f} |\n"
    md2 += f"| Cp | {np.mean([r['cp_r2'] for r in res_rel]):.4f} |\n"
    (REPORTS / "relative_geometry_study.md").write_text(md2)

    # WS3 & WS4: PointNet++ and Shootout V2
    print("[WS3/4] Neural Network LOAO and Shootout V2...")
    all_results = []
    all_results.append({'Model': 'LightGBM (Relative)', 'u_r2': mrel_u, 'wake_r2': np.mean([r['wake_r2'] for r in res_rel]), 'cp_r2': np.mean([r['cp_r2'] for r in res_rel])})
    
    # Random Forest Relative
    print("  -> RandomForest")
    rf_res = []
    for h in archs:
        tr, te = A_all != h, A_all == h
        rf_res.append(train_tabular(lambda: RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1), X_all[tr][:, idx_rel], Y_all[tr], X_all[te][:, idx_rel], Y_all[te]))
    all_results.append({'Model': 'RandomForest (Relative)', 'u_r2': np.mean([r['u_r2'] for r in rf_res]), 'wake_r2': np.mean([r['wake_r2'] for r in rf_res]), 'cp_r2': np.mean([r['cp_r2'] for r in rf_res])})

    # NNs
    best_nn_u = -999
    best_err_u, best_err_wake = None, None
    
    for cls, name in [(PointNetPP, 'PointNet++'), (EdgeGAT, 'EdgeGAT'), (TransV2, 'TransV2')]:
        print(f"  -> {name}")
        cls_res = []
        for h in archs:
            tr = [g for g in graphs if g.archetype != h]
            te = [g for g in graphs if g.archetype == h]
            r, fp, wp, yt = train_nn(cls, tr, te, epochs=15)
            cls_res.append(r)
            if name == 'PointNet++' and r['u_r2'] > best_nn_u:
                best_nn_u = r['u_r2']
                best_err_u = np.abs(yt[:,0] - fp[:,0])
                best_err_wake = np.abs(yt[:,4] - wp)
                best_te = te
        
        m_u = np.mean([r['u_r2'] for r in cls_res])
        m_w = np.mean([r['wake_r2'] for r in cls_res])
        m_c = np.mean([r['cp_r2'] for r in cls_res])
        all_results.append({'Model': name, 'u_r2': m_u, 'wake_r2': m_w, 'cp_r2': m_c})
        
        if name == 'PointNet++':
            md3 = f"# PointNet++ LOAO Results\n\n| Vel R² | Wake R² | Cp R² |\n|---|---|---|\n| {m_u:.4f} | {m_w:.4f} | {m_c:.4f} |\n"
            (REPORTS / "pointnet_loao.md").write_text(md3)

    res_df = pd.DataFrame(all_results).sort_values('u_r2', ascending=False)
    md4 = "# Representation Shootout V2\n\n| Model | Vel R² | Wake R² | Cp R² |\n|---|---|---|---|\n"
    for _, r in res_df.iterrows():
        md4 += f"| {r['Model']} | {r['u_r2']:.4f} | {r['wake_r2']:.4f} | {r['cp_r2']:.4f} |\n"
    (REPORTS / "representation_shootout_v2.md").write_text(md4)

    # WS5: Error Localization V2
    print("[WS5] Error Localization V2...")
    md5 = "# Error Localization V2 (PointNet++)\n\n"
    if best_err_u is not None:
        pos = np.vstack([g.pos.numpy() for g in best_te])
        z_mask_low = pos[:, 2] < 10
        md5 += f"| Region | Mean Absolute Error (u) |\n|---|---|\n"
        md5 += f"| Global | {np.mean(best_err_u):.4f} |\n"
        md5 += f"| Z < 10m | {np.mean(best_err_u[z_mask_low]):.4f} |\n"
    (REPORTS / "error_localization_v2.md").write_text(md5)

    # WS6: Phase Gate
    print("[WS6] Phase Gate Evaluation...")
    best_model = res_df.iloc[0]
    u, w, c = best_model['u_r2'], best_model['wake_r2'], best_model['cp_r2']
    
    if u > 0.5 and w > 0.5 and c > 0.3: cert, cert_str = "A", "GO for Deployment"
    elif u > 0.3 and w > 0.3 and c > 0: cert, cert_str = "B", "GO for Pilot"
    else: cert, cert_str = "C", "NO-GO (Below Certification B)"

    md6 = f"# Phase 8P Representation Replacement\n\n"
    md6 += f"**CERTIFICATION LEVEL: {cert}**\n**DECISION: {cert_str}**\n"
    md6 += f"**RECOMMENDED SURROGATE ARCHITECTURE**: {best_model['Model']}\n\n"
    
    md6 += f"## Answers\n"
    md6 += f"1. **Is coordinate leakage real?** {'YES' if (mb_u - mn_u) > 0.05 else 'NO'}\n"
    md6 += f"2. **Does relative geometry improve transferability?** {'YES' if mrel_u > mb_u else 'NO'}\n"
    md6 += f"3. **Does PointNet++ outperform LightGBM?** {'YES' if res_df[res_df['Model']=='PointNet++']['u_r2'].values[0] > res_df[res_df['Model']=='LightGBM (Relative)']['u_r2'].values[0] else 'NO'}\n"
    md6 += f"4. **Does PointNet++ outperform GNNs?** {'YES' if res_df[res_df['Model']=='PointNet++']['u_r2'].values[0] > res_df[res_df['Model']=='TransV2']['u_r2'].values[0] else 'NO'}\n"
    md6 += f"5. **Is Phase 9 deployment justified?** {'YES' if cert in ['A','B'] else 'NO'}\n"
    
    (REPORTS / "phase8p_representation_replacement.md").write_text(md6)
    print(f"\nPhase 8P Complete. Cert {cert}. Winner: {best_model['Model']} (u={u:.3f}, cp={c:.3f})")

if __name__ == "__main__":
    main()
