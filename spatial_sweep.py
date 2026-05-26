"""
spatial_sweep.py
================
Thesis: Multi-Scale Spatial Differential Analysis for Detecting Brain Injury
        Beyond Cortical Layers in Multiplex Tissue Imaging
Author: Razan Yousef (rkyousef)
Institution: University of Houston, Biomedical Engineering

Description:
    Parameter optimization sweep for spatial weight tuning in the joint
    sham/vehicle GMM clustering pipeline.

    Problem:
        Initial clustering used high spatial weight (10x), over-emphasizing
        geography. Low weight (0.5x) produced scattered phenotype clusters
        without spatial coherence.

    Solution:
        Sweep spatial weights [2.0, 5.0, 10.0], fit joint GMM on vehicle
        (br10) and sham (br24), compute differential enrichment, save figures.

    Result:
        Spatial weight = 5.0 produced the best balance, revealing an
        asymmetric injury-associated cluster in the vehicle brain absent
        in sham.

Marker Panel (16 channels):
    NeuN, S100, IBA1, RECA1, NFH, CC3, MBP, PCNA, MAP2, GAD67,
    GFAP, Parvalbumin, Calretinin, TomatoLectin, CD31, IBA1++

Usage:
    python spatial_sweep.py

Outputs (in ~/new2/thesis_results/spatial_sweep/):
    vehicle_sw{2.0,5.0,10.0}.png     -- vehicle brain cluster maps
    comparison_sw{2.0,5.0,10.0}.png  -- side-by-side vehicle vs sham
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

OUT = os.path.expanduser('~/new2/thesis_results/spatial_sweep/')
os.makedirs(OUT, exist_ok=True)

MARKERS = [
    'NeuN', 'S100', 'IBA1', 'RECA1', 'NFH', 'CC3', 'MBP', 'PCNA',
    'MAP2', 'GAD67', 'GFAP', 'Parvalbumin', 'Calretinin',
    'TomatoLectin', 'CD31', 'IBA1++'
]
VEH_PATH  = os.path.expanduser('~/new2/Datasets/vechile/br10.csv')
SHAM_PATH = os.path.expanduser('~/new2/Datasets/sham/br24.csv')
SPATIAL_WEIGHTS = [2.0, 5.0, 10.0]
N_COMPONENTS = 12


def build_features(df, spatial_weight):
    """
    Build feature matrix: marker intensities + z-normalized coordinates.
    Higher spatial_weight = geography dominates clustering.
    Lower spatial_weight = phenotype dominates clustering.
    Optimal: spatial_weight=3.0 (used in final multiscale framework).
    """
    Xm = df[MARKERS].values.astype(float)
    coords = df[['centroid_x', 'centroid_y']].values.astype(float)
    coords_z = (coords - coords.mean(0)) / coords.std(0) * spatial_weight
    return np.hstack([Xm, coords_z])


def compute_differential(lv, ls):
    """Fisher exact test + log2FC for each cluster vs sham."""
    results = []
    for c in sorted(np.unique(np.concatenate([lv, ls]))):
        nv = (lv == c).sum()
        ns = (ls == c).sum()
        fv = nv / len(lv)
        fs = ns / len(ls)
        lfc = np.log2((fv + 1e-8) / (fs + 1e-8))
        if abs(lfc) > 0.5:
            _, p = stats.fisher_exact(
                [[nv, len(lv) - nv], [ns, len(ls) - ns]]
            )
            results.append({
                'cluster': c, 'n_veh': nv, 'n_sham': ns,
                'log2FC': lfc, 'p': p,
                'direction': 'VEH' if lfc > 0 else 'SHAM'
            })
    return results


def main():
    print("Loading data...")
    veh  = pd.read_csv(VEH_PATH,  low_memory=False).dropna(subset=MARKERS)
    sham = pd.read_csv(SHAM_PATH, low_memory=False).dropna(subset=MARKERS)
    print(f"  Vehicle: {len(veh):,}  Sham: {len(sham):,}")

    for sw in SPATIAL_WEIGHTS:
        print(f"\nSpatial weight = {sw}")
        Xj = np.vstack([build_features(veh, sw), build_features(sham, sw)])
        gmm = GaussianMixture(
            n_components=N_COMPONENTS, covariance_type='full',
            n_init=3, max_iter=200, random_state=42
        )
        labels = gmm.fit_predict(Xj)
        lv, ls = labels[:len(veh)], labels[len(veh):]

        for d in compute_differential(lv, ls):
            sig = '*' if d['p'] < 0.05 else ' '
            print(f"  C{d['cluster']:>2}: log2FC={d['log2FC']:+.2f} "
                  f"p={d['p']:.1e}{sig} {d['direction']}")

        # Vehicle brain map
        fig, ax = plt.subplots(figsize=(14, 14))
        sc = ax.scatter(veh['centroid_y'], veh['centroid_x'],
                        c=lv, cmap='tab20', s=0.3, alpha=0.5)
        ax.set_title(f'Vehicle br10 — SW={sw}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Lateral (um)'); ax.set_ylabel('Depth (um)')
        plt.colorbar(sc, ax=ax, label='Cluster', shrink=0.7)
        plt.tight_layout()
        plt.savefig(OUT + f'vehicle_sw{sw}.png', dpi=200, bbox_inches='tight')
        plt.close()

        # Side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(28, 14))
        for ax, df_p, lab, ttl in [
            (axes[0], veh, lv, 'Vehicle br10'),
            (axes[1], sham, ls, 'Sham br24')
        ]:
            ax.scatter(df_p['centroid_y'], df_p['centroid_x'],
                       c=lab, cmap='tab20', s=0.3, alpha=0.5)
            ax.set_title(ttl, fontsize=16, fontweight='bold')
            ax.set_xlabel('Lateral'); ax.set_ylabel('Depth')
        plt.suptitle(f'Spatial Weight = {sw}', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUT + f'comparison_sw{sw}.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved sw={sw} figures.")

    print(f"Done. Check: {OUT}")


if __name__ == '__main__':
    main()
