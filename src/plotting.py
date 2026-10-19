import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_arcface_robustness(csv_path: str, output_dir: str, threshold: float = 0.4):
    """Generates standardized physical robustness trend plots from CSV data."""
    df = pd.read_csv(csv_path)
    phases = df['Phase'].unique()
    colors = {'No Glasses': 'blue', 'Black Glasses': 'orange', 'Adversarial Glasses': 'red'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Black-Box Physical Robustness Boundaries (ArcFace)', fontsize=16)

    for phase in phases:
        subset = df[df['Phase'] == phase]
        if subset.empty: continue
        
        sizes, offsets, sims = subset['Face_Width_px'], subset['Center_Offset_px'], subset['ArcFace_Similarity']
        
        ax1.scatter(sizes, sims, alpha=0.25, color=colors.get(phase, 'gray'))
        ax2.scatter(offsets, sims, alpha=0.25, color=colors.get(phase, 'gray'))
        
        # Quadratic trendlines
        p_size = np.poly1d(np.polyfit(sizes, sims, 2))
        x_sz = np.linspace(sizes.min(), sizes.max(), 100)
        ax1.plot(x_sz, p_size(x_sz), color=colors.get(phase, 'gray'), linewidth=2, label=phase)
        
        p_offset = np.poly1d(np.polyfit(offsets, sims, 2))
        x_off = np.linspace(offsets.min(), offsets.max(), 100)
        ax2.plot(x_off, p_offset(x_off), color=colors.get(phase, 'gray'), linewidth=2, label=phase)

    for ax, xlabel in zip([ax1, ax2], ['Face Bounding Box Width (Pixels)', 'Distance from Frame Center (Pixels)']):
        ax.axhline(y=threshold, color='k', linestyle='--', label='Detection Threshold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cosine Similarity (ArcFace)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plot_path = os.path.join(output_dir, "arcface_profiling_trends.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Trend plot exported to: {plot_path}")