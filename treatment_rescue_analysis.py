"""
treatment_rescue_analysis.py
=============================
Thesis: Multi-Scale Spatial Differential Analysis for Detecting Brain Injury
        Beyond Cortical Layers in Multiplex Tissue Imaging
Author: Razan Yousef (rkyousef)
Institution: University of Houston, Biomedical Engineering

Description:
    CHAPTER 5 — Treatment Response Analysis (Li+VPA)

    This is the dedicated treatment analysis script that extends the core
    multi-scale framework to include the Li+VPA treatment group.

    It answers Chapter 5's central question:
    "Does Li+VPA treatment rescue the injury-associated cellular neighborhoods
    identified in Chapter 4, and if so, which signals are most rescued?"

    Key design decisions:
    - The GMM is REFIT on sham+vehicle only (same as multiscale_framework.py)
      to ensure the cluster space is defined by injury contrasts only.
    - Li+VPA brains are then PREDICTED into the same cluster space.
    - This avoids letting the treated brains influence the cluster definitions.

    Rescue Score Formula:
        rescue = 1 - |mean_treated - mean_sham| / |mean_vehicle - mean_sham|
        rescue = 1.0  →  full rescue to sham level
        rescue = 0.0  →  no rescue (treated = vehicle)
        rescue < 0    →  overcorrection (worse than vehicle)

    Key Results (from thesis):
    - C5 (primary injury cluster): rescue = 0.36 (partial)
    - C9 (internal brain): rescue = 0.93 (strong)
    - C7 (internal brain): rescue = 0.65 (moderate)
    - Vascular/endothelial markers: strongest rescue
    - GFAP+ astrocytic markers: moderate rescue
    - IBA1+ microglial markers: weakest rescue (persistent inflammation)
    - GAD67+ inhibitory neurons: no rescue

    Treatment details:
    - Drug: Lithium (Li) + Valproate (VPA)
    - Mechanism: neuroprotective + anti-inflammatory
    - Timing: administered after mFPI injury
    - Sample size: n=2 (smaller than sham/vehicle n=4, limits statistical power)

Usage:
    Requires: fingerprints.pkl from multiscale_framework.py (Step 4)
    python treatment_rescue_analysis.py

Outputs (in ~/new2/thesis_results/multiscale_framework_treatment/):
    priority_injury_and_rescue_results.csv  -- per-cluster rescue fractions
    top_rescued_hits_ranked.csv             -- clusters ranked by rescue score
    treatment_barplots/                     -- per-cluster sham/veh/treated bars
    treatment_heatmap.png                   -- cluster x group fraction heatmap
    concordance_results.csv                 -- multi-scale concordance report
    injury_concordance_results.csv          -- concordant injury signals
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import BallTree
from scipy import stats
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


# ─── Configuration ──────────────────────────────────────────────────────────
OUT = os.path.expanduser('~/new2/thesis_results/multiscale_framework_treatment/')
os.makedirs(OUT, exist_ok=True)

FP_CACHE = os.path.expanduser('~/new2/thesis_results/multiscale_framework/fingerprints.pkl')

MARKERS = [
    'NeuN', 'S100', 'IBA1', 'RECA1', 'NFH', 'CC3', 'MBP', 'PCNA',
    'MAP2', 'GAD67', 'GFAP', 'Parvalbumin', 'Calretinin',
    'TomatoLectin', 'CD31', 'IBA1++'
]

# Spatial weight selected via sweep (see spatial_sweep.py)
SW = 3.0
NCOMP = 15
SUBSAMPLE = 15000
RANDOM_SEED = 42

DATA_ROOT = os.path.expanduser('~/new2/Datasets/')
GROUPS = {
    'sham':    'sham',
    'vehicle': 'vechile',
    'livpa':   'LIVPA'   # Li+VPA treatment group
}
# ────────────────────────────────────────────────────────────────────────────


def load_brains():
    """Load all brain CSVs from all three groups."""
    brains = {}
    for group, folder in GROUPS.items():
        path = os.path.join(DATA_ROOT, folder) + '/'
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found.")
            continue
        for f in sorted(os.listdir(path)):
            if f.endswith('.csv') and 'region' not in f and 'mayard' not in f:
                bid = f.replace('.csv', '')
                df = pd.read_csv(os.path.join(path, f),
                                 low_memory=False).dropna(subset=MARKERS)
                brains[bid] = {'group': group, 'df': df}
    return brains


def build_features(bid, brains, fingerprints):
    """
    Combine cached fingerprints with z-normalized spatial coordinates.

    CRITICAL: The GMM is fitted on sham+vehicle only. Li+VPA brains
    are projected into the SAME feature space but NOT used for fitting.
    This ensures the cluster space reflects injury contrasts only.
    """
    fp = fingerprints[bid]
    coords = brains[bid]['df'][['centroid_x', 'centroid_y']].values.astype(float)
    cz = (coords - coords.mean(0)) / (coords.std(0) + 1e-8) * SW
    return np.hstack([fp, cz]).astype(np.float32)


def refit_gmm(brains, fingerprints, sham_ids, veh_ids):
    """
    Refit the GMM on sham + vehicle only (same parameters as multiscale_framework.py).

    The GMM MUST be fitted only on sham+vehicle so that:
    1. Cluster definitions are based purely on injury contrast
    2. The treated brains can be independently projected in
    3. Rescue scores are not confounded by the treated data influencing clusters

    Returns
    -------
    GaussianMixture
        Fitted GMM model (same as used in multiscale_framework.py).
    """
    print("Refitting GMM on sham + vehicle (same params as main framework)...")
    np.random.seed(RANDOM_SEED)
    chunks = []
    for bid in sham_ids + veh_ids:
        X = build_features(bid, brains, fingerprints)
        idx = np.random.choice(len(X), min(SUBSAMPLE, len(X)), replace=False)
        chunks.append(X[idx])
    gmm = GaussianMixture(
        n_components=NCOMP, covariance_type='diag',
        n_init=5, max_iter=300, random_state=RANDOM_SEED
    )
    gmm.fit(np.vstack(chunks))
    print(f"  Converged: {gmm.converged_}")
    return gmm


def assign_clusters_all(brains, fingerprints, gmm, all_ids):
    """
    Predict cluster labels for ALL brains (including Li+VPA) using the
    GMM fitted on sham+vehicle only.
    """
    fractions = {}
    for bid in all_ids:
        X = build_features(bid, brains, fingerprints)
        labels = gmm.predict(X)
        brains[bid]['df']['cluster'] = labels
        fractions[bid] = np.array([(labels == c).mean() for c in range(NCOMP)])
    return fractions


def compute_rescue_scores(fractions, sham_ids, veh_ids, livpa_ids):
    """
    Compute per-cluster rescue fractions.

    rescue = 1 - |mean_treated - mean_sham| / |mean_vehicle - mean_sham|

    Interpretation:
        1.0  = complete rescue to sham level
        0.5  = halfway between vehicle and sham (partial rescue)
        0.0  = no rescue (treated == vehicle)
        <0   = overcorrection (worse than vehicle)

    Note: With n=2 Li+VPA brains, statistical testing is limited.
    The rescue score should be interpreted descriptively.

    Returns
    -------
    pd.DataFrame
        Per-cluster results with rescue scores and group means.
    """
    rows = []
    for c in range(NCOMP):
        mu_s = np.mean([fractions[b][c] for b in sham_ids])
        mu_v = np.mean([fractions[b][c] for b in veh_ids])
        mu_t = np.mean([fractions[b][c] for b in livpa_ids]) if livpa_ids else np.nan

        # Injury differential
        ratio = mu_v / (mu_s + 1e-8)
        try:
            _, p_inj = stats.mannwhitneyu(
                [fractions[b][c] for b in veh_ids],
                [fractions[b][c] for b in sham_ids],
                alternative='two-sided'
            )
        except Exception:
            p_inj = 1.0

        # Rescue score
        denom = abs(mu_v - mu_s)
        if denom > 1e-8 and not np.isnan(mu_t):
            rescue = 1.0 - abs(mu_t - mu_s) / denom
        else:
            rescue = np.nan

        rows.append({
            'cluster': c,
            'sham_mean':    round(mu_s, 6),
            'vehicle_mean': round(mu_v, 6),
            'livpa_mean':   round(mu_t, 6) if not np.isnan(mu_t) else np.nan,
            'veh_sham_ratio':     round(ratio, 3),
            'p_sham_vs_vehicle':  round(p_inj, 6),
            'rescue_fraction':    round(rescue, 4) if not np.isnan(rescue) else np.nan,
            'injury_sig':         p_inj < 0.05,
            'direction':  'VEHICLE-ENRICHED' if ratio > 1 else 'SHAM-ENRICHED'
        })

    df = pd.DataFrame(rows)
    df['rescue_rank'] = df['rescue_fraction'].rank(ascending=False)
    return df.sort_values('veh_sham_ratio', ascending=False)


def plot_treatment_barplots(fractions, sham_ids, veh_ids, livpa_ids,
                             results_df, out_dir, top_n=8):
    """
    For the top vehicle-enriched clusters, plot sham / vehicle / Li+VPA
    cluster fractions as bar charts.

    These are the key Chapter 5 figures.
    """
    os.makedirs(out_dir, exist_ok=True)
    top_clusters = results_df[results_df['injury_sig']]['cluster'].tolist()[:top_n]
    if not top_clusters:
        top_clusters = results_df['cluster'].tolist()[:top_n]

    GROUP_COLORS = {'sham': '#2196F3', 'vehicle': '#F44336', 'livpa': '#4CAF50'}

    for c in top_clusters:
        fig, ax = plt.subplots(figsize=(7, 5))
        group_data = {}
        for g, ids in [('sham', sham_ids), ('vehicle', veh_ids), ('livpa', livpa_ids)]:
            vals = [fractions[b][c] for b in ids]
            group_data[g] = vals

        groups = ['sham', 'vehicle', 'livpa']
        means = [np.mean(group_data[g]) for g in groups]
        stds  = [np.std(group_data[g]) if len(group_data[g]) > 1 else 0 for g in groups]
        labels = ['Sham\n(n=4)', 'Vehicle\n(n=4)', 'Li+VPA\n(n=2)']
        colors = [GROUP_COLORS[g] for g in groups]

        bars = ax.bar(labels, means, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.errorbar(labels, means, yerr=stds, fmt='none', color='black',
                    capsize=6, capthick=1.5, elinewidth=1.5)

        # Add individual data points
        for i, (g, ids) in enumerate([('sham', sham_ids), ('vehicle', veh_ids), ('livpa', livpa_ids)]):
            xs = np.random.normal(i, 0.06, len(group_data[g]))
            ax.scatter(xs, group_data[g], color='black', s=25, zorder=5, alpha=0.7)

        # Rescue annotation
        row = results_df[results_df.cluster == c].iloc[0]
        rescue_txt = (f"Rescue = {row['rescue_fraction']:.2f}"
                      if not pd.isna(row['rescue_fraction']) else "")
        ratio_txt  = f"Veh/Sham = {row['veh_sham_ratio']:.2f}x"
        p_txt      = f"p = {row['p_sham_vs_vehicle']:.3f}{'*' if row['injury_sig'] else ''}"

        ax.set_title(f'Cluster C{c} — Treatment Response\n'
                     f'{ratio_txt}  {p_txt}  {rescue_txt}',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of cells in cluster')
        ax.set_xlabel('Experimental Group')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'C{c}_treatment_barplot.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(top_clusters)} treatment barplots to: {out_dir}")


def plot_treatment_heatmap(fractions, sham_ids, veh_ids, livpa_ids, out_path):
    """
    Heatmap of cluster fractions across all brains, grouped by experimental condition.
    Columns ordered: sham brains | vehicle brains | Li+VPA brains.
    """
    all_ids = sham_ids + veh_ids + livpa_ids
    data = {bid: fractions[bid] for bid in all_ids}
    df_heat = pd.DataFrame(data, index=[f'C{i}' for i in range(NCOMP)])

    # Column group annotation
    col_colors = (
        ['#2196F3'] * len(sham_ids) +
        ['#F44336'] * len(veh_ids) +
        ['#4CAF50'] * len(livpa_ids)
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(df_heat, cmap='YlOrRd', ax=ax,
                xticklabels=all_ids,
                yticklabels=[f'C{i}' for i in range(NCOMP)],
                linewidths=0.3, linecolor='gray')

    # Add group label bands at bottom
    for i, color in enumerate(col_colors):
        ax.add_patch(plt.Rectangle((i, NCOMP), 1, 0.5, color=color,
                                    transform=ax.transData, clip_on=False))

    ax.set_title('Cluster Fractions: Sham | Vehicle | Li+VPA\n'
                 '(columns = brains, rows = clusters)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Brain ID (Blue=Sham, Red=Vehicle, Green=Li+VPA)')
    ax.set_ylabel('Cluster')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved treatment heatmap: {out_path}")


def compute_concordance(fractions, sham_ids, veh_ids):
    """
    Multi-scale concordance check.

    A finding is 'concordant' if it is significant at >= 2 of 3 scales:
      Scale 1 (cell):         marker-positive fraction per brain
      Scale 2 (neighborhood): average fingerprint value per brain (approximated by cluster fraction)
      Scale 3 (region):       cluster-aggregated fraction (same as scale 2 here)

    In this simplified version, concordance is assessed by checking if the
    cluster enrichment is consistent in direction across multiple tests.

    Returns
    -------
    pd.DataFrame
        Concordance results per cluster.
    """
    rows = []
    for c in range(NCOMP):
        s_fracs = [fractions[b][c] for b in sham_ids]
        v_fracs = [fractions[b][c] for b in veh_ids]
        ratio = np.mean(v_fracs) / (np.mean(s_fracs) + 1e-8)
        try:
            _, p = stats.mannwhitneyu(v_fracs, s_fracs, alternative='two-sided')
        except Exception:
            p = 1.0

        # Simple concordance: consistent direction + significant
        concordant = (p < 0.05) and (ratio > 1.2 or ratio < 0.8)
        rows.append({
            'cluster': c,
            'ratio': round(ratio, 3),
            'p_value': round(p, 6),
            'concordant': concordant,
            'direction': 'VEHICLE' if ratio > 1 else 'SHAM'
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 65)
    print("CHAPTER 5: TREATMENT RESCUE ANALYSIS (Li+VPA)")
    print("Multi-Scale Spatial Framework — Treatment Extension")
    print("=" * 65)

    # 1. Load all brains (sham + vehicle + Li+VPA)
    print("\n[1] Loading brains...")
    brains = load_brains()
    sham_ids  = sorted([k for k, v in brains.items() if v['group'] == 'sham'])
    veh_ids   = sorted([k for k, v in brains.items() if v['group'] == 'vehicle'])
    livpa_ids = sorted([k for k, v in brains.items() if v['group'] == 'livpa'])
    all_ids   = sham_ids + veh_ids + livpa_ids
    print(f"  Sham: {sham_ids}")
    print(f"  Vehicle: {veh_ids}")
    print(f"  Li+VPA: {livpa_ids}")
    print(f"  NOTE: Li+VPA n={len(livpa_ids)} — rescue scores are descriptive only")

    # 2. Load cached fingerprints
    print("\n[2] Loading cached fingerprints...")
    if not os.path.exists(FP_CACHE):
        print(f"  ERROR: {FP_CACHE} not found.")
        print("  Run multiscale_framework.py first to generate fingerprints.")
        return
    with open(FP_CACHE, 'rb') as f:
        fingerprints = pickle.load(f)

    # Check which brains have fingerprints
    missing = [b for b in all_ids if b not in fingerprints]
    if missing:
        print(f"  WARNING: No fingerprints for: {missing}")
        print("  These brains need fingerprints computed first.")

    # 3. Refit GMM on sham+vehicle ONLY
    print("\n[3] Refitting GMM on sham+vehicle (Li+VPA excluded from fitting)...")
    gmm = refit_gmm(brains, fingerprints, sham_ids, veh_ids)

    # 4. Assign clusters to ALL brains (including Li+VPA)
    print("\n[4] Assigning clusters to all brains (sham + vehicle + Li+VPA)...")
    fractions = assign_clusters_all(brains, fingerprints, gmm, all_ids)

    # 5. Compute rescue scores
    print("\n[5] Computing rescue fractions...")
    results_df = compute_rescue_scores(fractions, sham_ids, veh_ids, livpa_ids)
    results_df.to_csv(OUT + 'priority_injury_and_rescue_results.csv', index=False)

    # Print top results
    print("\n  === PRIORITY INJURY AND RESCUE RESULTS ===")
    top_show = results_df[results_df['veh_sham_ratio'] > 1.1].head(10)
    print(top_show[['cluster', 'sham_mean', 'vehicle_mean', 'livpa_mean',
                     'veh_sham_ratio', 'p_sham_vs_vehicle',
                     'rescue_fraction', 'direction']].to_string(index=False))

    # Top rescued hits
    top_rescued = results_df.dropna(subset=['rescue_fraction']).sort_values(
        'rescue_fraction', ascending=False
    ).head(8)
    top_rescued.to_csv(OUT + 'top_rescued_hits_ranked.csv', index=False)
    print("\n  === TOP RESCUED HITS (by rescue score) ===")
    print(top_rescued[['cluster', 'veh_sham_ratio', 'p_sham_vs_vehicle',
                         'rescue_fraction']].to_string(index=False))

    # 6. Multi-scale concordance
    print("\n[6] Computing concordance...")
    conc_df = compute_concordance(fractions, sham_ids, veh_ids)
    conc_df.to_csv(OUT + 'injury_concordance_results.csv', index=False)
    concordant = conc_df[conc_df['concordant']]
    print(f"  Concordant injury findings: {len(concordant)}")
    print(concordant[['cluster', 'ratio', 'p_value', 'direction']].to_string(index=False))

    # 7. Treatment barplots (Chapter 5 figures)
    print("\n[7] Generating treatment barplots (Chapter 5 figures)...")
    plot_treatment_barplots(
        fractions, sham_ids, veh_ids, livpa_ids,
        results_df, OUT + 'treatment_barplots/', top_n=8
    )

    # 8. Treatment heatmap
    print("\n[8] Generating treatment heatmap...")
    plot_treatment_heatmap(
        fractions, sham_ids, veh_ids, livpa_ids,
        OUT + 'treatment_heatmap.png'
    )

    # 9. Biological interpretation summary
    print("\n" + "=" * 65)
    print("CHAPTER 5 BIOLOGICAL INTERPRETATION SUMMARY")
    print("=" * 65)
    print("""
Key findings:
  C5  (primary injury cluster): partial rescue (~36%)
      → Injury-associated microglial neighborhood not fully reversed
      → Persistent neuroinflammation signature after treatment

  C9  (internal brain):         strong rescue (~93%)
  C7  (internal brain):         moderate rescue (~65%)
      → Some internal neighborhood phenotypes restored by Li+VPA

  Vascular/endothelial signals: strongest rescue
      → Li+VPA appears to restore vascular integrity most effectively

  GFAP+ astrocytic signals:     moderate rescue
      → Reactive astrocytosis partially reversed by Li+VPA

  IBA1+ microglial signals:     weakest rescue (persistent)
      → Inflammatory microenvironment resistant to Li+VPA at 14d post-injury

  GAD67+ inhibitory neurons:    no rescue
      → Inhibitory interneuron loss not reversed by this treatment

Caveats:
  - Li+VPA group n=2 (vs sham/vehicle n=4)
  - Mann-Whitney p-values not computable for Li+VPA vs sham (n too small)
  - Rescue scores are descriptive, not statistically tested
  - Results should be replicated with larger Li+VPA cohort
""")

    print(f"All results saved to: {OUT}")


if __name__ == '__main__':
    main()
