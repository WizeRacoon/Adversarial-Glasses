import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_digital_histogram(sim_scores, threshold, mean_sim, success_rate, plot_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sim_scores, bins=35, color='royalblue', edgecolor='black', alpha=0.75)
    ax.axvline(threshold, color='crimson', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax.axvline(mean_sim, color='black', linestyle='-', linewidth=2, label=f'Mean ({mean_sim:.2f})')
    ax.set_title(f'Digital Robustness Simulation (Success Rate: {success_rate:.1f}%)')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

def plot_single_experiment_trends(csv_path, sim_col, threshold, model_name, output_path):
    """Generates the dual scatter/trend plots for a single evaluation run."""
    df = pd.read_csv(csv_path)
    colors = {'No Glasses': 'blue', 'Black Glasses': 'orange', 'Adversarial Glasses': 'red'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Physical Robustness Boundaries ({model_name})', fontsize=16)

    for phase in df['Phase'].unique():
        phase_data = df[df['Phase'] == phase]
        sizes = phase_data['Face_Width_px'].values
        offsets = phase_data['Center_Offset_px'].values
        sims = phase_data[sim_col].values

        # Scatter points
        ax1.scatter(sizes, sims, alpha=0.25, color=colors.get(phase, 'k'), edgecolors='none')
        ax2.scatter(offsets, sims, alpha=0.25, color=colors.get(phase, 'k'), edgecolors='none')

        # Polynomial Trendlines
        if len(sizes) > 1:
            idx_sizes = np.argsort(sizes)
            p_size = np.poly1d(np.polyfit(sizes, sims, 2))
            ax1.plot(sizes[idx_sizes], p_size(sizes[idx_sizes]), color=colors.get(phase, 'k'), linewidth=2, label=phase)

            idx_offsets = np.argsort(offsets)
            p_offset = np.poly1d(np.polyfit(offsets, sims, 2))
            ax2.plot(offsets[idx_offsets], p_offset(offsets[idx_offsets]), color=colors.get(phase, 'k'), linewidth=2, label=phase)

    ax1.axhline(y=threshold, color='k', linestyle='--', label='Detection Threshold')
    ax1.set_xlabel('Face Bounding Box Width (Pixels)')
    ax1.set_ylabel(f'Cosine Similarity ({model_name})')
    ax1.legend()

    ax2.axhline(y=threshold, color='k', linestyle='--', label='Detection Threshold')
    ax2.set_xlabel('Distance from Frame Center (Pixels)')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def generate_ablation_plots(baseline_csv, exp_csvs: dict, threshold, output_dir, model_name="ResNet", sim_col="ResNet_Similarity"):
    """
    Generates the final report plots: Aggregate Boxplot and Ablation Trendlines.
    exp_csvs = {"Exp 1 (Ideal)": "path1.csv", "Exp 2 (Phone)": "path2.csv", ...}
    """
    sns.set_theme(style="whitegrid")
    
    # 1. Build Aggregate DataFrame
    df_baseline = pd.read_csv(baseline_csv)
    df_combined = df_baseline[df_baseline['Phase'].isin(['No Glasses', 'Black Glasses'])].copy()
    
    for exp_label, csv_path in exp_csvs.items():
        if os.path.exists(csv_path):
            df_exp = pd.read_csv(csv_path)
            df_adv = df_exp[df_exp['Phase'] == 'Adversarial Glasses'].copy()
            df_adv['Phase'] = exp_label 
            df_combined = pd.concat([df_combined, df_adv], ignore_index=True)

    # Reorder for the boxplot layout
    order = ['No Glasses', 'Black Glasses'] + list(exp_csvs.keys())

    # --- Plot A: Aggregate Boxplot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Phase', y=sim_col, data=df_combined, order=order, ax=ax, palette="Set2")
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax.set_title(f'Aggregate Physical Evasion Robustness ({model_name})', fontsize=14)
    ax.set_ylabel('Cosine Similarity')
    ax.set_xlabel('Ablation Phase')
    plt.xticks(rotation=15)
    ax.legend()
    fig.tight_layout()
    box_path = os.path.join(output_dir, f"aggregate_boxplot_{model_name.lower()}.png")
    fig.savefig(box_path, dpi=300)
    plt.close(fig)

    # --- Plot B: Combined Trendlines ---
    # Only scatter the baselines, map polynomial lines for all experiments
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Comparative Physical Robustness Boundaries ({model_name})', fontsize=16)

    colors = sns.color_palette("husl", len(order))
    color_map = dict(zip(order, colors))

    for phase in order:
        phase_data = df_combined[df_combined['Phase'] == phase]
        if phase_data.empty:
            continue

        sizes = phase_data['Face_Width_px'].values
        offsets = phase_data['Center_Offset_px'].values
        sims = phase_data[sim_col].values

        # Scatter ONLY for baselines
        if phase in ['No Glasses', 'Black Glasses']:
            ax1.scatter(sizes, sims, alpha=0.15, color=color_map[phase], edgecolors='none')
            ax2.scatter(offsets, sims, alpha=0.15, color=color_map[phase], edgecolors='none')

        # Polynomial line for ALL
        if len(sizes) > 1:
            idx_sizes = np.argsort(sizes)
            p_size = np.poly1d(np.polyfit(sizes, sims, 2))
            ax1.plot(sizes[idx_sizes], p_size(sizes[idx_sizes]), color=color_map[phase], linewidth=2.5, label=phase)

            idx_offsets = np.argsort(offsets)
            p_offset = np.poly1d(np.polyfit(offsets, sims, 2))
            ax2.plot(offsets[idx_offsets], p_offset(offsets[idx_offsets]), color=color_map[phase], linewidth=2.5, label=phase)

    ax1.axhline(y=threshold, color='k', linestyle='--', label='Threshold')
    ax1.set_xlabel('Face Bounding Box Width (Pixels)')
    ax1.set_ylabel('Cosine Similarity')
    ax1.legend(loc='lower left', fontsize=9)

    ax2.axhline(y=threshold, color='k', linestyle='--', label='Threshold')
    ax2.set_xlabel('Distance from Frame Center (Pixels)')
    ax2.legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    trend_path = os.path.join(output_dir, f"robustness_trendlines_{model_name.lower()}.png")
    fig.savefig(trend_path, dpi=300)
    plt.close(fig)

    print(f"Generated aggregate boxplot at: {box_path}")
    print(f"Generated combined trendlines at: {trend_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Plotting Utilities")
    parser.add_argument("--mode", type=str, required=True, choices=["ablation", "single"], help="Plot generation mode")
    parser.add_argument("--baseline_csv", type=str, help="Path to baseline robustness CSV (provides No Glasses & Black Glasses data)")
    parser.add_argument("--exp1", type=str, help="Path to Exp 1 CSV")
    parser.add_argument("--exp2", type=str, help="Path to Exp 2 CSV")
    parser.add_argument("--exp3", type=str, help="Path to Exp 3 CSV")
    parser.add_argument("--exp4", type=str, help="Path to Exp 4 CSV")
    parser.add_argument("--out_dir", type=str, default="../outputs", help="Output directory for charts")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "ablation":
        if not args.baseline_csv:
            print("Error: --baseline_csv is required for ablation mode.")
            exit(1)
        
        # Maps the CLI arguments to your reporting layout
        experiments = {}
        if args.exp1: experiments["Exp 1 (Ideal)"] = args.exp1
        if args.exp2: experiments["Exp 2 (Phone)"] = args.exp2
        if args.exp3: experiments["Exp 3 (25 Similar)"] = args.exp3
        if args.exp4: experiments["Exp 4 (25 Diverse)"] = args.exp4

        generate_ablation_plots(args.baseline_csv, experiments, threshold=0.3, 
                                output_dir=args.out_dir, model_name="ResNet", sim_col="ResNet_Similarity")
        generate_ablation_plots(args.baseline_csv, experiments, threshold=0.4, 
                                output_dir=args.out_dir, model_name="ArcFace", sim_col="ArcFace_Similarity")