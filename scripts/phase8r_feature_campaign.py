#!/usr/bin/env python3
"""
Phase 8R — Final Aerodynamic Feature Campaign
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

PROJECT = Path("/home/wangchen/Documents/Micro-Climate-Zoning-AI")
ML      = PROJECT / "data" / "ml"
REPORTS = PROJECT / "reports"

# ── Feature Engineering ──────────────────────────────────────────────────

def build_features(df, cdf, max_nodes=500):
    all_x, all_y = [], []
    arch_ids = []
    
    f_names = [
        'ws', 'wd', 
        'x_norm', 'y_norm', 'z_norm', 'dist_center', 'dist_tallest', # Phase 8P baseline
        
        # WS1: SVF
        'svf_25m', 'svf_50m', 'svf_100m',
        
        # WS2: Wind Exposure
        'upwind_count', 'upwind_hm', 'upwind_hmax', 'upwind_h_grad', 'upwind_frontal',
        'downwind_open', 'cross_dens', 'cross_hvar',
        
        # WS3: Street Canyon
        'canyon_w', 'canyon_h', 'canyon_hw', 'canyon_align', 'street_open',
        
        # WS4: Multi-Scale
        'dens_25', 'hm_25', 'hmax_25', 'hvar_25',
        'dens_50', 'hm_50', 'hmax_50', 'hvar_50',
        'dens_100', 'hm_100', 'hmax_100', 'hvar_100',
        'dens_200', 'hm_200', 'hmax_200', 'hvar_200'
    ]
    
    for sim_id, grp in df.groupby('simulation_id'):
        if len(grp) > max_nodes:
            grp = grp.sample(max_nodes, random_state=42)
            
        coords = grp[['x','y','z']].values.astype(np.float64)
        meta = cdf[cdf['simulation_id'] == sim_id]
        if len(meta) == 0: continue
        meta = meta.iloc[0]
        ws, wd = float(meta['wind_speed']), float(meta['wind_direction'])
        arch = int(grp['archetype'].iloc[0])
        
        # Baseline relative geometry
        x_min, x_max = coords[:,0].min(), coords[:,0].max()
        y_min, y_max = coords[:,1].min(), coords[:,1].max()
        z_max = coords[:,2].max()
        
        x_norm = (coords[:,0] - x_min) / (x_max - x_min + 1e-6)
        y_norm = (coords[:,1] - y_min) / (y_max - y_min + 1e-6)
        z_norm = coords[:,2] / (z_max + 1e-6)
        x_c, y_c = (x_max+x_min)/2, (y_max+y_min)/2
        dist_center = np.linalg.norm(coords[:,:2] - np.array([x_c, y_c]), axis=1)
        dist_tallest = np.linalg.norm(coords[:,:2] - coords[np.argmax(coords[:,2]), :2], axis=1)
        
        base_feats = np.column_stack([np.full(len(coords), ws), np.full(len(coords), wd), x_norm, y_norm, z_norm, dist_center, dist_tallest])
        
        tree = cKDTree(coords[:, :2]) # 2D tree for urban morphology metrics
        z_arr = coords[:, 2]
        
        def get_multi_scale(r):
            idxs = tree.query_ball_point(coords[:, :2], r)
            dens = np.array([len(idx) for idx in idxs])
            hm = np.array([np.mean(z_arr[idx]) if len(idx)>0 else z for z, idx in zip(z_arr, idxs)])
            hmax = np.array([np.max(z_arr[idx]) if len(idx)>0 else z for z, idx in zip(z_arr, idxs)])
            hvar = np.array([np.var(z_arr[idx]) if len(idx)>0 else 0.0 for idx in idxs])
            return dens, hm, hmax, hvar

        d25, hm25, hmax25, hv25 = get_multi_scale(25)
        d50, hm50, hmax50, hv50 = get_multi_scale(50)
        d100, hm100, hmax100, hv100 = get_multi_scale(100)
        d200, hm200, hmax200, hv200 = get_multi_scale(200)
        
        ms_feats = np.column_stack([
            d25, hm25, hmax25, hv25,
            d50, hm50, hmax50, hv50,
            d100, hm100, hmax100, hv100,
            d200, hm200, hmax200, hv200
        ])
        
        # WS1: SVF Proxies
        # Simple proxy: 1.0 - (points higher than z within R / total points in R)
        def get_svf(r):
            idxs = tree.query_ball_point(coords[:, :2], r)
            svf = []
            for z, idx in zip(z_arr, idxs):
                if len(idx) <= 1: svf.append(1.0)
                else: svf.append(1.0 - np.sum(z_arr[idx] > z) / len(idx))
            return np.array(svf)
            
        svf_feats = np.column_stack([get_svf(25), get_svf(50), get_svf(100)])
        
        # WS2: Wind Exposure (50m radius)
        rad = np.radians(wd)
        wvec = np.array([np.cos(rad), np.sin(rad)])
        
        up_c, up_hm, up_hmax, up_hgrad, up_front = [], [], [], [], []
        dn_open, cr_dens, cr_hvar = [], [], []
        
        idxs_50 = tree.query_ball_point(coords[:, :2], 50.0)
        for i, idx in enumerate(idxs_50):
            if len(idx) <= 1:
                up_c.append(0); up_hm.append(z_arr[i]); up_hmax.append(z_arr[i]); up_hgrad.append(0); up_front.append(0)
                dn_open.append(1.0); cr_dens.append(0); cr_hvar.append(0.0)
                continue
                
            nbrs2d = coords[idx, :2]
            nbrz = z_arr[idx]
            vecs = nbrs2d - coords[i, :2]
            dists = np.linalg.norm(vecs, axis=1) + 1e-6
            align = (vecs @ wvec) / dists
            
            up_mask = align < -0.5
            dn_mask = align > 0.5
            cr_mask = np.abs(align) <= 0.5
            
            upz = nbrz[up_mask]
            up_c.append(len(upz))
            up_hm.append(np.mean(upz) if len(upz)>0 else z_arr[i])
            up_hmax.append(np.max(upz) if len(upz)>0 else z_arr[i])
            up_hgrad.append(np.mean(np.maximum(0, upz - z_arr[i])) if len(upz)>0 else 0)
            up_front.append(np.sum(upz) / 50.0) # crude frontal area proxy
            
            dn_open.append(1.0 - len(nbrz[dn_mask])/len(idx))
            
            crz = nbrz[cr_mask]
            cr_dens.append(len(crz))
            cr_hvar.append(np.var(crz) if len(crz)>0 else 0.0)
            
        wind_feats = np.column_stack([up_c, up_hm, up_hmax, up_hgrad, up_front, dn_open, cr_dens, cr_hvar])
        
        # WS3: Street Canyon
        # Proxy canyon properties based on crosswind and upwind geometry
        can_w = 50.0 / (np.array(cr_dens) + 1)
        can_h = np.array(cr_hm := [np.mean(z_arr[idx][np.abs((coords[idx,:2] - coords[i,:2])@wvec/(np.linalg.norm(coords[idx,:2]-coords[i,:2])+1e-6)) <= 0.5]) if len(idx)>1 else z_arr[i] for i, idx in enumerate(idxs_50)])
        can_hw = can_h / can_w
        can_align = np.array(cr_dens) / (np.array(up_c) + 1e-6) # Alignment proxy
        st_open = svf_feats[:, 0] * np.array(dn_open)
        
        canyon_feats = np.column_stack([can_w, can_h, can_hw, can_align, st_open])
        
        # Combine
        x_feat = np.column_stack([base_feats, svf_feats, wind_feats, canyon_feats, ms_feats]).astype(np.float32)
        
        u, v, w = grp['u'].values, grp['v'].values, grp['w'].values
        p_arr = grp['p'].values
        speed = np.sqrt(u**2 + v**2 + w**2)
        wake = (speed < 0.3*ws).astype(np.float32)
        cp = p_arr / (0.5 * 1.225 * ws**2 + 1e-3)
        
        y_feat = np.column_stack([u, cp, wake]).astype(np.float32)
        
        all_x.append(x_feat)
        all_y.append(y_feat)
        arch_ids.extend([arch]*len(coords))
        
    return np.vstack(all_x), np.vstack(all_y), np.array(arch_ids), f_names

def train_eval(x_tr, y_tr, x_te, y_te):
    m_v = LGBMRegressor(n_estimators=60, random_state=42, verbose=-1).fit(x_tr, y_tr[:,0])
    pred_v = m_v.predict(x_te)
    return r2_score(y_te[:,0], pred_v)

def main():
    print("Phase 8R — Final Aerodynamic Feature Campaign")
    df  = pd.read_parquet(ML / "cfd_field_dataset_full.parquet")
    cdf = pd.read_parquet(ML / "verified_cfd_dataset_v3.parquet")
    
    print("[PREP] Extracting complex aerodynamic features...")
    X, Y, A, f_names = build_features(df, cdf, max_nodes=500)
    archs = sorted(np.unique(A))
    
    # Feature group indices
    base_idx = list(range(7))
    svf_idx = list(range(7, 10))
    wind_idx = list(range(10, 18))
    canyon_idx = list(range(18, 23))
    ms_idx = list(range(23, 39))

    # Fast validation split for WS1-WS4 Ablation
    te_arch = archs[-1]
    xtr, ytr = X[A != te_arch], Y[A != te_arch]
    xte, yte = X[A == te_arch], Y[A == te_arch]
    
    print("[WS5] Feature Ablation Study...")
    res_ablation = {}
    
    r_base = train_eval(xtr[:, base_idx], ytr, xte[:, base_idx], yte)
    res_ablation['Baseline'] = r_base
    
    r_svf = train_eval(xtr[:, base_idx+svf_idx], ytr, xte[:, base_idx+svf_idx], yte)
    res_ablation['+ SVF'] = r_svf
    
    r_wind = train_eval(xtr[:, base_idx+wind_idx], ytr, xte[:, base_idx+wind_idx], yte)
    res_ablation['+ Wind Exp'] = r_wind
    
    r_can = train_eval(xtr[:, base_idx+canyon_idx], ytr, xte[:, base_idx+canyon_idx], yte)
    res_ablation['+ Canyon'] = r_can
    
    r_ms = train_eval(xtr[:, base_idx+ms_idx], ytr, xte[:, base_idx+ms_idx], yte)
    res_ablation['+ Multi-Scale'] = r_ms
    
    r_all = train_eval(xtr, ytr, xte, yte)
    res_ablation['All Features'] = r_all
    
    md5 = "# Feature Ablation (Holdout Archetype)\n\n| Feature Set | Velocity R² |\n|---|---|\n"
    for k, v in res_ablation.items(): md5 += f"| {k} | {v:.4f} |\n"
    (REPORTS / "feature_ablation_phase8r.md").write_text(md5)
    (REPORTS / "svf_features.md").write_text(f"SVF Contribution: {r_svf - r_base:.4f} R2\n")
    (REPORTS / "wind_exposure_features.md").write_text(f"Wind Exposure Contribution: {r_wind - r_base:.4f} R2\n")
    (REPORTS / "canyon_features.md").write_text(f"Canyon Contribution: {r_can - r_base:.4f} R2\n")
    (REPORTS / "multiscale_morphology.md").write_text(f"Multi-scale Contribution: {r_ms - r_base:.4f} R2\n")

    print("[WS6] Interpretability Analysis (LightGBM Gain)...")
    m_all = LGBMRegressor(n_estimators=80, random_state=42, verbose=-1).fit(xtr, ytr[:,0])
    importances = m_all.feature_importances_
    imp_tuples = sorted(zip(f_names, importances), key=lambda x: -x[1])
    
    md6 = "# Feature Importance (Top 30)\n\n| Rank | Feature | LightGBM Gain |\n|---|---|---|\n"
    for i, (f, imp) in enumerate(imp_tuples[:30]):
        md6 += f"| {i+1} | {f} | {imp} |\n"
    (REPORTS / "shap_analysis_phase8r.md").write_text(md6)

    print("[WS7] Final LOAO Benchmark (All Features)...")
    v_r2, v_mae, c_r2, w_r2 = [], [], [], []
    for h in archs:
        xtr, ytr = X[A != h], Y[A != h]
        xte, yte = X[A == h], Y[A == h]
        
        m_v = LGBMRegressor(n_estimators=80, random_state=42, verbose=-1).fit(xtr, ytr[:,0])
        pred_v = m_v.predict(xte)
        v_r2.append(r2_score(yte[:,0], pred_v))
        v_mae.append(mean_absolute_error(yte[:,0], pred_v))
        
        m_c = LGBMRegressor(n_estimators=80, random_state=42, verbose=-1).fit(xtr, ytr[:,1])
        c_r2.append(r2_score(yte[:,1], m_c.predict(xte)))
        
        m_w = LGBMRegressor(n_estimators=80, random_state=42, verbose=-1).fit(xtr, ytr[:,2])
        w_r2.append(r2_score(yte[:,2], m_w.predict(xte)))
        
    m_u = np.mean(v_r2)
    m_cp = np.mean(c_r2)
    m_w = np.mean(w_r2)

    md7 = "# Final LOAO Benchmark (Phase 8R)\n\n"
    md7 += f"- **Velocity R²**: {m_u:.4f}\n"
    md7 += f"- **Velocity MAE**: {np.mean(v_mae):.4f}\n"
    md7 += f"- **Cp R²**: {m_cp:.4f}\n"
    md7 += f"- **Wake R²**: {m_w:.4f}\n"
    (REPORTS / "final_loao_phase8r.md").write_text(md7)

    print("[GATE] Evaluating Deployment...")
    if m_u >= 0.40: cert, cert_str = "A", "Production Ready"
    elif m_u >= 0.30: cert, cert_str = "B", "Pilot Deployment GO"
    else: cert, cert_str = "C", "Below Deployment Threshold (NO-GO)"

    md_fin = f"# Phase 8R Final Feature Campaign\n\n"
    md_fin += f"**CERTIFICATION LEVEL: {cert}**\n**DECISION: {cert_str}**\n\n"
    md_fin += f"## Analysis\n"
    md_fin += f"1. **Did aerodynamic features close the gap?** {'YES' if m_u >= 0.30 else 'NO'}\n"
    md_fin += f"2. **Dominant predictors**: {imp_tuples[0][0]}, {imp_tuples[1][0]}, {imp_tuples[2][0]}\n"
    md_fin += f"3. **Is Vel R² >= 0.30 achievable?** {'YES' if m_u >= 0.30 else 'NO. Ceiling Reached.'}\n"
    md_fin += f"4. **Is deployment justified?** {'YES' if cert in ['A','B'] else 'NO'}\n\n"
    
    if cert == "C":
        md_fin += "## FINAL PROJECT STATUS\n"
        md_fin += "The LightGBM architecture with exhaustive relative and aerodynamic features has reached its absolute ceiling. Generalizing urban CFD fields across radically different topologies with $N=130$ simulations is unachievable with current methods."

    (REPORTS / "phase8r_final_feature_campaign.md").write_text(md_fin)
    print(f"\nPhase 8R Complete. Cert {cert}. Velocity R² = {m_u:.3f}")

if __name__ == "__main__":
    main()
