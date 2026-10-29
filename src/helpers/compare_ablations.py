import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adjusted path to step out of helpers/ and src/
OUTPUT_BASE_DIR = "../../outputs"

EXPERIMENT_FOLDERS = {
    "exp1_single_ideal": "Exp 1 (Single Ideal)",
    "exp2_single_phone": "Exp 2 (Single Phone)",
    "exp3_25_similar": "Exp 3 (25 Similar)",
    "exp4_25_diverse": "Exp 4 (25 Diverse)"
}

def generate_comparison():
    data_rows = []
    
    first_folder = list(EXPERIMENT_FOLDERS.keys())[0]
    first_exp_path = os.path.join(OUTPUT_BASE_DIR, first_folder, "dual_robustness.csv")
    
    if not os.path.exists(first_exp_path):
        raise FileNotFoundError(f"Could not find baseline data at {first_exp_path}")
        
    df_base = pd.read_csv(first_exp_path)
    for phase in ['No Glasses', 'Black Glasses']:
        for _, row in df_base[df_base['Phase'] == phase].iterrows():
            data_rows.append({'Condition': phase, 'Model': 'ResNet', 'Similarity': row['ResNet_Similarity']})
            data_rows.append({'Condition': phase, 'Model': 'ArcFace', 'Similarity': row['ArcFace_Similarity']})

    for folder_name, plot_label in EXPERIMENT_FOLDERS.items():
        csv_path = os.path.join(OUTPUT_BASE_DIR, folder_name, "dual_robustness.csv")
        if os.path.exists(csv_path):
            df_exp = pd.read_csv(csv_path)
            for _, row in df_exp[df_exp['Phase'] == 'Adversarial Glasses'].iterrows():
                condition_name = f"Adv: {plot_label}"
                data_rows.append({'Condition': condition_name, 'Model': 'ResNet', 'Similarity': row['ResNet_Similarity']})
                data_rows.append({'Condition': condition_name, 'Model': 'ArcFace', 'Similarity': row['ArcFace_Similarity']})
        else:
            print(f"Warning: {csv_path} not found. Skipping...")

    df_all = pd.DataFrame(data_rows)
    
    sns.set_theme(style="whitegrid")
    out_dir = os.path.join(OUTPUT_BASE_DIR, "final_charts")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- RESNET PLOT ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_all[df_all['Model'] == 'ResNet'], x='Condition', y='Similarity', palette="crest")
    plt.axhline(0.3, ls='--', color='black', label='Detection Threshold (0.3)')
    plt.title('ResNet (White-Box) Evasion', fontsize=14, fontweight='bold')
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    resnet_out = os.path.join(out_dir, "ablation_comparison_resnet.png")
    plt.savefig(resnet_out, dpi=300, bbox_inches='tight')
    plt.close()

    # --- ARCFACE PLOT ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_all[df_all['Model'] == 'ArcFace'], x='Condition', y='Similarity', palette="flare")
    plt.axhline(0.4, ls='--', color='black', label='Detection Threshold (0.4)')
    plt.title('ArcFace (Black-Box) Transferability', fontsize=14, fontweight='bold')
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    arcface_out = os.path.join(out_dir, "ablation_comparison_arcface.png")
    plt.savefig(arcface_out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSuccess! Separated charts saved to:\n  - {resnet_out}\n  - {arcface_out}")

if __name__ == "__main__":
    generate_comparison()