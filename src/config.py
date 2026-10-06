import os
import torch

# Paths
DATA_DIR = os.path.join("..", "data", "sample")
OUTPUT_DIR = os.path.join("..", "outputs")
MASK_TEMPLATE_PATH = os.path.join("..", "data", "glasses_template.png")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimization Hyperparameters
ITERATIONS = 500
LEARNING_RATE = 0.01
BATCH_SIZE = 5

# Loss Weights
WEIGHT_TV = 0.001
WEIGHT_NPS = 0.0025
WEIGHT_DODGE = 1.0

# EoT Parameters
AFFINE_DEGREES = 15
AFFINE_TRANSLATE = (0.08, 0.08)
AFFINE_SCALE = (0.9, 1.1)