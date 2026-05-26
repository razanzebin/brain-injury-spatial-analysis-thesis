"""
visualize_br10.py
=================
Thesis: Multi-Scale Spatial Differential Analysis for Detecting Brain Injury
        Beyond Cortical Layers in Multiplex Tissue Imaging
        Author: Razan Yousef (rkyousef)
        Institution: University of Houston, Biomedical Engineering

        Description:
            Loads AVDGP clustering results from a .mat file and the corresponding
                cell coordinates from a CSV, then renders a high-resolution spatial map
                    of cluster assignments projected onto the brain's anatomical axes.
                        This was the initial visualization that revealed the "accidental eureka":
                            the unsupervised clustering had isolated an injury-associated region
                                extending into deeper brain layers.

                                Usage:
                                    python visualize_br10.py

                                    Outputs:
                                        ~/new2/AVDGP_Autonomous_Delineation/results/br10_map_clean.png
                                        """

import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os


def load_coordinates(csv_path):
      """Load X/Y cell centroid coordinates from the brain CSV dataset."""
      df = pd.read_csv(csv_path, usecols=[1, 2], low_memory=False)
      return df


def load_cluster_assignments(mat_path):
      """Load AVDGP cluster label assignments from the HDF5-formatted .mat file."""
      with h5py.File(mat_path, 'r') as f:
                clusters = np.array(f['assign']).flatten().astype(int)
            return clusters


def main():
      # --- Paths ---
      csv_path = os.path.expanduser(
          '~/new2/Datasets/vechile/br10.csv'
)
    mat_path = os.path.expanduser(
              '~/new2/AVDGP_Autonomous_Delineation/results/br10_results.mat'
    )
    out_path = os.path.expanduser(
              '~/new2/AVDGP_Autonomous_Delineation/results/br10_map_clean.png'
    )

    # --- Load data ---
    print("Loading coordinates...")
    df = load_coordinates(csv_path)

    print("Loading AVDGP cluster assignments...")
    clusters = load_cluster_assignments(mat_path)
    df['cluster'] = clusters

    # --- Plot ---
    # Wide figure to stretch the brain laterally
    plt.figure(figsize=(18, 24))

    # X = cortical depth (vertical axis), Y = lateral axis (horizontal)
    plt.scatter(
              df.iloc[:, 1],   # lateral axis
              df.iloc[:, 0],   # cortical depth
              c=df['cluster'],
              s=0.8,
              cmap='tab20',
              alpha=0.9
    )

    plt.colorbar(label='Cluster ID')
    plt.title('Brain 10: High-Res Anatomical Delineation (AVDGP Clustering)')

    # Zoom into tissue region to remove whitespace
    plt.xlim(10000, 42000)

    plt.xlabel('Lateral Axis (um)')
    plt.ylabel('Cortical Depth (um)')
    plt.grid(False)

    # --- Save ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Zoomed map saved to: {out_path}")


if __name__ == '__main__':
      main()
