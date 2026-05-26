"""
cortex_internal_analysis.py
============================
Thesis: Multi-Scale Spatial Differential Analysis for Detecting Brain Injury
        Beyond Cortical Layers in Multiplex Tissue Imaging
Author: Razan Yousef (rkyousef)
Institution: University of Houston, Biomedical Engineering

Description:
    Beyond-cortex compartment analysis: separates the brain tissue into
    (1) cortex-band and (2) internal brain regions, then compares cluster
    fractions, marker profiles, and treatment rescue across compartments.

    This directly addresses the central thesis question:
    "Do injury-associated cellular neighborhoods extend BEYOND the cortex
    into internal brain structures?"

    Method:
        The cortex is identified as cells within a depth band near the
        tissue surface (using morphological image processing on the cell
        density grid). All remaining cells are classified as internal brain.
        Cluster enrichment is then compared between the two compartments
        across sham, vehicle, and Li+VPA groups.

    Key Result:
        Cluster C5 shows 3.38x enrichment in the CORTEX-BAND and 3.58x
        enrichment in the INTERNAL BRAIN (both p=0.029), confirming that
        the injury-associated neighborhood extends beyond the cortical rim
        into deeper structures.

        This is the central finding of the thesis.

Usage:
    Requires multiscale_framework.py to have been run first
    (needs fingerprints.pkl and cluster assignments).

    python cortex_internal_analysis.py

Outputs (in ~/new2/thesis_results/cortex_internal_analysis/):
    cluster_cortex_internal_heatmap.png
    cluster_cortex_internal_stats.csv
    ranked_cortex_clusters.csv
    ranked_internal_clusters.csv
    C5_cortex_internal_barplots.png
    top_internal_cluster_marker_heatmap.png
    top_internal_cluster_marker_profiles.csv
    top_internal_cluster_barplots.png
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.morphology import binary_closing, binary_opening, disk
from scipy import stats


# ─── Configuration ─────────────────────────────────────────────────────────
OUT = os.path.expanduser('~/new2/thesis_results/cortex_internal_analysis/')
os.makedirs(OUT, exist_ok=True)

FP_DIR = os.path.expanduser('~/new2/thesis_results/multiscale_framework/')

MARKERS = [
    'NeuN', 'S100', 'IBA1', 'RECA1', 'NFH', 'CC3', 'MBP', 'PCNA',
    'MAP2', 'GAD67', 'GFAP', 'Parvalbumin', 'Calretinin',
    'TomatoLectin', 'CD31', 'IBA1++'
]
NCOMP = 15
SW    = 3.0
SUBSAMPLE = 15000
RANDOM_SEED = 42

DATA_ROOT = os.path.expanduser('~/new2/Datasets/')
GROUPS = {
    'sham':    'sham',
    'vehicle': 'vechile',
    'livpa':   'LIVPA'
}
# ───────────────────────────────────────────────────────────────────────────


def load_brains():
    """Load all brain CSVs."""
    brains = {}
    for group, folder in GROUPS.items():
        path = os.path.join(DATA_ROOT, folder) + '/'
        if not os.path.exists(path):
            continue
        for f in sorted(os.listdir(path)):
            if f.endswith('.csv') and 'region' not in f and 'mayard' not in f:
                bid = f.replace('.csv', '')
                df = pd.read_csv(os.path.join(path, f),
                                 low_memory=False).dropna(subset=MARKERS)
                brains[bid] = {'group': group, 'df': df.copy()}
    return brains


def assign_cortex_band(df, grid_res=500, cortex_depth_quantile=0.20):
    """
    Identify cortex-band cells using a density-based surface detection.

    Approach:
        1. Bin cells into a 2D spatial grid.
        2. Create a binary occupancy map (True where cells exist).
        3. Apply morphological closing to fill gaps.
        4. Use binary_fill_holes to close the interior.
        5. Compute distance transform from tissue edge.
        6. Cells within the outer depth_quantile fraction are labeled
           as cortex-band; the rest are internal brain.

    Parameters
    ----------
    df : pd.DataFrame
        Cell dataframe with centroid_x, centroid_y.
    grid_res : int
        Grid resolution in micrometers.
    cortex_depth_quantile : float
        Fraction of tissue depth to consider cortex (default: 20%).

    Returns
    -------
    pd.Series
        'cortex_band' or 'internal_brain' labels per cell.
    """
    x = df['centroid_x'].values
    y = df['centroid_y'].values

    # Build occupancy grid
    x_bins = np.arange(x.min(), x.max() + grid_res, grid_res)
    y_bins = np.arange(y.min(), y.max() + grid_res, grid_res)
    H, xedge, yedge = np.histogram2d(x, y, bins=[x_bins, y_bins])
    occupied = H > 0

    # Morphological operations to define tissue boundary
    occupied = binary_closing(occupied, disk(2))
    occupied = binary_fill_holes(occupied)

    # Distance from edge (interior cells have higher distance)
    dist = distance_transform_edt(occupied)
    max_dist = dist.max()
    cortex_threshold = max_dist * cortex_depth_quantile

    # Assign each cell to a compartment
    xi = np.clip(np.digitize(x, xedge) - 1, 0, H.shape[0] - 1)
    yi = np.clip(np.digitize(y, yedge) - 1, 0, H.shape[1] - 1)
    cell_dist = dist[xi, yi]

    labels = np.where(cell_dist <= cortex_threshold, 'cortex_band', 'internal_brain')
    return pd.Series(labels, index=df.index)


def build_features(df, fingerprints, bid):
    """Combine fingerprint with z-normalized coordinates."""
    fp = fingerprints[bid]
    coords = df[['centroid_x', 'centroid_y']].values.astype(float)
    cz = (coords - coords.mean(0)) / (coords.std(0) + 1e-8) * SW
    return np.hstack([fp, cz]).astype(np.float32)


def main():
    print("=" * 60)
    print("CORTEX-BAND vs INTERNAL BRAIN COMPARTMENT ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n[1] Loading brains...")
    brains = load_brains()
    sham_ids  = sorted([k for k, v in brains.items() if v['group'] == 'sham'])
    veh_ids   = sorted([k for k, v in brains.items() if v['group'] == 'vehicle'])
    livpa_ids = sorted([k for k, v in brains.items() if v['group'] == 'livpa'])
    all_ids   = sham_ids + veh_ids + livpa_ids

    # Load fingerprints
    print("\n[2] Loading cached fingerprints...")
    fp_cache = FP_DIR + 'fingerprints.pkl'
    with open(fp_cache, 'rb') as f:
        fingerprints = pickle.load(f)

    # Fit GMM (same as multiscale_framework.py)
    print("\n[3] Fitting joint GMM...")
    np.random.seed(RANDOM_SEED)
    chunks = []
    for bid in sham_ids + veh_ids:
        X = build_features(brains[bid]['df'], fingerprints, bid)
        idx = np.random.choice(len(X), min(SUBSAMPLE, len(X)), replace=False)
        chunks.append(X[idx])
    gmm = GaussianMixture(n_components=NCOMP, covariance_type='diag',
                          n_init=5, max_iter=300, random_state=RANDOM_SEED)
    gmm.fit(np.vstack(chunks))

    # Predict clusters + assign compartments
    print("\n[4] Predicting clusters and assigning compartments...")
    for bid in all_ids:
        df = brains[bid]['df']
        X = build_features(df, fingerprints, bid)
        df['cluster'] = gmm.predict(X)
        df['region'] = assign_cortex_band(df)
        brains[bid]['df'] = df

    # Compute per-brain per-compartment cluster fractions
    print("\n[5] Computing compartment cluster fractions...")
    rows = []
    for bid in all_ids:
        df = brains[bid]['df']
        group = brains[bid]['group']
        for region in ['cortex_band', 'internal_brain']:
            sub = df[df['region'] == region]
            if len(sub) == 0:
                continue
            for c in range(NCOMP):
                frac = (sub['cluster'] == c).mean()
                rows.append({'brain': bid, 'group': group,
                             'region': region, 'cluster': c, 'fraction': frac})
    frac_df = pd.DataFrame(rows)

    # Differential analysis per compartment
    print("\n[6] Computing differential enrichment per compartment...")
    stat_rows = []
    for region in ['cortex_band', 'internal_brain']:
        for c in range(NCOMP):
            s_fracs = frac_df[(frac_df.group == 'sham') &
                              (frac_df.region == region) &
                              (frac_df.cluster == c)]['fraction'].values
            v_fracs = frac_df[(frac_df.group == 'vehicle') &
                              (frac_df.region == region) &
                              (frac_df.cluster == c)]['fraction'].values
            t_fracs = frac_df[(frac_df.group == 'livpa') &
                              (frac_df.region == region) &
                              (frac_df.cluster == c)]['fraction'].values

            if len(s_fracs) < 2 or len(v_fracs) < 2:
                continue

            ratio = np.mean(v_fracs) / (np.mean(s_fracs) + 1e-8)
            try:
                _, p = stats.mannwhitneyu(v_fracs, s_fracs, alternative='two-sided')
            except Exception:
                p = 1.0

            mu_s = np.mean(s_fracs)
            mu_v = np.mean(v_fracs)
            mu_t = np.mean(t_fracs) if len(t_fracs) > 0 else np.nan
            denom = abs(mu_v - mu_s)
            rescue = (1.0 - abs(mu_t - mu_s) / denom
                      if denom > 1e-8 and not np.isnan(mu_t) else np.nan)
            stat_rows.append({
                'cluster': c,
                'region': region,
                'sham_mean': round(mu_s, 6),
                'vehicle_mean': round(mu_v, 6),
                'livpa_mean': round(mu_t, 6),
                'veh_sham_ratio': round(ratio, 3),
                'p_sham_vs_vehicle': round(p, 6),
                'rescue_fraction': round(rescue, 4) if not np.isnan(rescue) else np.nan
            })

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(OUT + 'cluster_cortex_internal_stats.csv', index=False)

    # Print top vehicle-enriched clusters
    print("\n  === TOP INTERNAL VEHICLE-ENRICHED CLUSTERS ===")
    top_internal = (stats_df[stats_df.region == 'internal_brain']
                    .sort_values('veh_sham_ratio', ascending=False)
                    .head(8))
    print(top_internal[['cluster', 'region', 'sham_mean', 'vehicle_mean',
                         'livpa_mean', 'veh_sham_ratio',
                         'p_sham_vs_vehicle', 'rescue_fraction']].to_string(index=False))

    # Save ranked tables
    for region in ['cortex_band', 'internal_brain']:
        ranked = (stats_df[stats_df.region == region]
                  .sort_values('veh_sham_ratio', ascending=False))
        fname = 'ranked_cortex_clusters.csv' if region == 'cortex_band' else 'ranked_internal_clusters.csv'
        ranked.to_csv(OUT + fname, index=False)

    # --- Figure 1: Heatmap of cluster fractions by compartment ---
    pivot = stats_df.pivot_table(
        values='veh_sham_ratio', index='cluster', columns='region'
    ).fillna(1.0)
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot, cmap='RdYlBu_r', center=1.0, ax=ax,
                annot=True, fmt='.2f', linewidths=0.5)
    ax.set_title('Vehicle/Sham Ratio by Cluster and Compartment\n'
                 '(>1 = vehicle-enriched; red = injury-associated)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT + 'cluster_cortex_internal_heatmap.png', dpi=240, bbox_inches='tight')
    plt.close()
    print("\n  Saved: cluster_cortex_internal_heatmap.png")

    # --- Figure 2: C5 barplots by compartment ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, region in enumerate(['cortex_band', 'internal_brain']):
        ax = axes[i]
        group_means = {}
        for g in ['sham', 'vehicle', 'livpa']:
            vals = frac_df[(frac_df.group == g) & (frac_df.region == region) &
                           (frac_df.cluster == 5)]['fraction'].values
            group_means[g] = (np.mean(vals), np.std(vals) if len(vals) > 1 else 0)
        groups = list(group_means.keys())
        means = [group_means[g][0] for g in groups]
        stds  = [group_means[g][1] for g in groups]
        colors = ['#2196F3', '#F44336', '#4CAF50']
        ax.bar(groups, means, color=colors, alpha=0.8, edgecolor='black')
        ax.errorbar(groups, means, yerr=stds, fmt='none', color='black', capsize=5)
        ax.set_title(f'C5 in {region.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylabel('Fraction of cells in cluster')
        ax.set_xlabel('Group')
    plt.suptitle('Cluster C5: The Primary Injury-Associated Neighborhood\n'
                 'Extends into BOTH Cortex-Band and Internal Brain',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT + 'C5_cortex_internal_barplots.png', dpi=240, bbox_inches='tight')
    plt.close()
    print("  Saved: C5_cortex_internal_barplots.png")

    # --- Figure 3: Top internal cluster marker heatmap ---
    top5_clusters = (stats_df[stats_df.region == 'internal_brain']
                     .sort_values('veh_sham_ratio', ascending=False)
                     .head(5)['cluster'].tolist())
    all_df = pd.concat([brains[b]['df'] for b in all_ids
                        if 'cluster' in brains[b]['df'].columns])
