"""
autonomous_delineator.py
========================
Thesis: Multi-Scale Spatial Differential Analysis for Detecting Brain Injury
        Beyond Cortical Layers in Multiplex Tissue Imaging
Author: Razan Yousef (rkyousef)
Institution: University of Houston, Biomedical Engineering

Description:
    Objective 2 of the thesis pipeline: Autonomous Anatomical Segmentation.
    Uses a Gaussian Mixture Model (GMM) on all numeric cell-level features
    (marker intensities + morphology) from the vehicle brain (br10) to
    unsupervisedly identify spatially meaningful anatomical regions.

    Key Finding:
    While this script was originally designed for cortical layer delineation,
    it produced the "accidental eureka" discovery: the GMM clustering
    identified a distinct spatial region with altered cellular composition
    that extends into deeper brain structures beyond the cortical layers,
    suggesting injury-associated abnormalities at subcortical depths.

    This result motivated the development of the full multi-scale spatial
    differential analysis framework (see spatial_sweep.py and
    multiscale_framework.py).

Usage:
    python autonomous_delineator.py

Outputs:
    ~/new2/AVDGP_Autonomous_Delineation/results/br10_anatomical_map.png

Dependencies:
    pandas, numpy, matplotlib, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os


# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = os.path.expanduser('~/new2/Datasets/vechile/br10.csv')
OUT_PATH  = os.path.expanduser(
    '~/new2/AVDGP_Autonomous_Delineation/results/br10_anatomical_map.png'
)
N_COMPONENTS = 8   # Number of anatomical regions / cluster components
RANDOM_STATE  = 42
# ──────────────────────────────────────────────────────────────────────────────


def load_and_prepare(csv_path):
    """
    Load the brain cell dataset and extract numeric features for clustering.

    Parameters
    ----------
    csv_path : str
        Path to the cell-level CSV file containing marker intensities,
        morphology, and centroid coordinates.

    Returns
    -------
    df : pd.DataFrame
        Full dataframe including centroid columns.
    features_scaled : np.ndarray
        Standardized numeric feature matrix (n_cells x n_features).
    """
    print(f"Loading Brain 10 from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Select all numeric columns; drop ID-like metadata columns
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=['ID', 'Unnamed: 0'], errors='ignore'
    ).copy()

    print(f"  Cells: {len(df):,}  |  Features used: {numeric_df.shape[1]}")

    # Standardize: ensures morphological features (e.g., Area in um^2)
    # are treated on the same scale as marker intensities
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(numeric_df)

    return df, features_scaled


def run_gmm(features_scaled, n_components=N_COMPONENTS, random_state=RANDOM_STATE):
    """
    Fit a Gaussian Mixture Model to the cell feature matrix.

    The GMM infers soft cluster memberships. Using n_components=8 captures
    the broad anatomical regions (cortex, striatum, corpus callosum, etc.)
    present in a coronal brain section.

    Parameters
    ----------
    features_scaled : np.ndarray
        Standardized feature matrix.
    n_components : int
        Number of mixture components (anatomical regions).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    labels : np.ndarray
        Hard cluster assignment per cell.
    """
    print(f"Segmenting into {n_components} anatomical regions via GMM...")
    gmm = GaussianMixture(
        n_components=n_components,
        reg_covar=1e-5,
        random_state=random_state
    )
    labels = gmm.fit_predict(features_scaled)
    print(f"  Clustering complete.")
    return labels


def plot_anatomical_map(df, labels, out_path):
    """
    Render the cluster assignments onto centroid_x / centroid_y coordinates.

    The centroid columns are the true spatial coordinates of each cell
    nucleus in the tissue image, so the scatter plot recreates the
    anatomical structure of the brain slice.

    Parameters
    ----------
    df : pd.DataFrame
        Cell dataframe with centroid_x and centroid_y columns.
    labels : np.ndarray
        Cluster labels per cell.
    out_path : str
        File path to save the figure.
    """
    plt.figure(figsize=(15, 10))

    scatter = plt.scatter(
        df['centroid_x'],
        df['centroid_y'],
        c=labels,
        s=0.5,
        cmap='nipy_spectral',
        alpha=0.8
    )

    plt.colorbar(scatter, label='Autonomous Layer ID')
    plt.title('Objective 2: Autonomous Anatomical Segmentation (Brain 10)\n'
              'GMM on all numeric cell features — reveals deeper injury zone')
    plt.xlabel('Lateral Axis (um)')
    plt.ylabel('Cortical Depth (um)')
    plt.gca().invert_yaxis()  # depth increases downward
    plt.grid(False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Anatomical map saved to: {out_path}")


def main():
    df, features_scaled = load_and_prepare(DATA_PATH)
    labels = run_gmm(features_scaled)
    df['layer_id'] = labels
    plot_anatomical_map(df, labels, OUT_PATH)


if __name__ == '__main__':
    main()
