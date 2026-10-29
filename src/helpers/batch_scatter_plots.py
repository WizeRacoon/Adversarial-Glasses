import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adjusted path to step out of helpers/ and src/
OUTPUT_BASE_DIR = "../../outputs"
EXPERIMENTS = [
    "exp1_single_ideal",
    "exp2_single_phone",
    "exp3_25_similar",
    "exp4_25_diverse"
]

COL_PHASE = "Phase"
COL_RESNET = "ResNet_Similarity"
COL_ARCFACE = "ArcFace_Similarity"
COL_WIDTH = "Face_Width_px"        
COL_DISTANCE = "Center_Offset_px"

COLORS = {
    "No Glasses": "blue",
    "Black Glasses": "orange",
    "Adversarial Glasses": "red"
}

def plot_robustness():
    sns.set_theme(style="white")
    
    for exp in EXPERIMENTS:
        csv_path = os.path.join(OUTPUT_BASE_DIR, exp, "dual_robustness.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: Data not found for {exp} at {csv_path}. Skipping.")
            continue
            
        df = pd.read_csv(csv_path)
        
        # --- RESNET PLOT ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Physical Robustness Boundaries (ResNet) - {exp}", fontsize=16)
        
        for phase, color in COLORS.items():
            subset = df[df[COL_PHASE] == phase]
            if not subset.empty:
                sns.regplot(data=subset, x=COL_WIDTH, y=COL_RESNET, ax=axes[0], color=color, 
                            label=phase, order=2, scatter_kws={'alpha':0.3, 'edgecolor':'none'})
                sns.regplot(data=subset, x=COL_DISTANCE, y=COL_RESNET, ax=axes[1], color=color, 
                            label=phase, order=2, scatter_kws={'alpha':0.3, 'edgecolor':'none'})
        
        axes[0].set_xlabel("Face Bounding Box Width (Pixels)")
        axes[0].set_ylabel("Cosine Similarity (ResNet)")
        axes[0].axhline(0.3, ls='--', color='black', label="Detection Threshold")
        axes[0].legend()
        
        axes[1].set_xlabel("Distance from Frame Center (Pixels)")
        axes[1].set_ylabel("")
        axes[1].axhline(0.3, ls='--', color='black', label="Detection Threshold")
        axes[1].legend()
        
        plt.tight_layout()
        resnet_out = os.path.join(OUTPUT_BASE_DIR, exp, f"resnet_boundaries.png")
        plt.savefig(resnet_out, dpi=300, bbox_inches='tight')
        plt.close()

        # --- ARCFACE PLOT ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Physical Robustness Boundaries (ArcFace) - {exp}", fontsize=16)
        
        for phase, color in COLORS.items():
            subset = df[df[COL_PHASE] == phase]
            if not subset.empty:
                sns.regplot(data=subset, x=COL_WIDTH, y=COL_ARCFACE, ax=axes[0], color=color, 
                            label=phase, order=2, scatter_kws={'alpha':0.3, 'edgecolor':'none'})
                sns.regplot(data=subset, x=COL_DISTANCE, y=COL_ARCFACE, ax=axes[1], color=color, 
                            label=phase, order=2, scatter_kws={'alpha':0.3, 'edgecolor':'none'})
        
        axes[0].set_xlabel("Face Bounding Box Width (Pixels)")
        axes[0].set_ylabel("Cosine Similarity (ArcFace)")
        axes[0].axhline(0.4, ls='--', color='black', label="Detection Threshold")
        axes[0].legend()
        
        axes[1].set_xlabel("Distance from Frame Center (Pixels)")
        axes[1].set_ylabel("")
        axes[1].axhline(0.4, ls='--', color='black', label="Detection Threshold")
        axes[1].legend()
        
        plt.tight_layout()
        arcface_out = os.path.join(OUTPUT_BASE_DIR, exp, f"arcface_boundaries.png")
        plt.savefig(arcface_out, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Successfully generated plots for {exp}")

if __name__ == "__main__":
    plot_robustness()