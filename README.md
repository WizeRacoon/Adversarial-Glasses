# Adversarial Eyeglass Framework

This repository serves as the codebase for an MSc research project focusing on physical adversarial evasion attacks against state-of-the-art Face Recognition Systems (FRSs). Building upon the foundational research of Sharif et al. in *"Accessorize to a Crime"*, this project reproduces, modernizes, and extends the physical realization of adversarial eyeglass frames. 

The repository provides a complete, differentiable pipeline for optimizing printable adversarial textures. It actively integrates Expectation over Transformation (EoT), Total Variation (TV) constraints, and a Non-Printability Score (NPS) to bridge the digital-to-physical performance gap.

**Current Scope & Active Development**
The primary focus of this internship is establishing a highly robust framework for **dodging** (untargeted evasion). The pipeline is currently being polished and expanded to maximize real-world physical robustness through:
* **Subject-Specific Robustness:** Implementing multi-image batching across diverse lighting conditions and poses to prevent spatial overfitting.
* **Ensemble-Based Optimization:** Structuring the pipeline to compute joint objective functions across both Euclidean (InceptionResnetV1) and Additive Angular Margin (ArcFace) architectures.

**Future Research Directions**
The codebase is designed with modularity in mind, leaving a structured foundation for future students and subsequent projects to explore:
* Targeted Impersonation Attacks
* Hardware Fluctuations and Morphological Impact (e.g., varying frame surface areas)
* Defensive Countermeasures and Mitigation

## Repository Structure

```text
Adversarial-Glasses/
├── data/
│   ├── sample/                   # Public/stock face images for baseline testing
│   ├── private/                  # Personal evaluation faces (IGNORED by Git)
│   └── glasses_template.png      # High-contrast RGBA eyeglass silhouette
├── notebooks/                    # Prototyping and custom exploratory analyses
├── outputs/                      # Generated PDFs, evaluation plots, and textures
├── src/
│   ├── config.py                 # Pipeline hyperparameters and constants
│   ├── eot.py                    # Expectation over Transformation augmentations
│   ├── evaluate.py               # Digital simulation and live webcam testing
│   ├── losses.py                 # Total Variation, NPS, and Cosine Dodging losses
│   ├── models.py                 # Model wrappers (InceptionResnetV1, MTCNN)
│   ├── optimize.py               # Optimization loop and batch driver
│   └── utils.py                  # Masking routines, inpainting, and PDF rendering
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup & Installation

1. **Activate Environment:**
   Ensure Conda is installed and create the project environment:
   ```bash
   conda activate adv_glasses
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Operating Instructions

### 1. Data Preparation
- Place target face images into `data/sample/` (for public testing) or `data/private/` (for personal testing).
- Keep the transparent mask file at `data/glasses_template.png`.
- *Note:* The repository's `.gitignore` rules prevent any media inside `data/private/` or stray image files from being tracked by Git.

### 2. Generating Adversarial Glasses
To run the optimization pipeline:
```bash
cd src
python optimize.py
```
- The optimizer calculates joint loss (Cosine Dodging, Total Variation for smoothness, and Non-Printability Score for color realizability).
- When optimization finishes, a high-resolution, vector-scaled PDF with cutting contours is exported to:
  `outputs/physical_adversarial_glasses.pdf`

### 3. Hyperparameter Configuration
Modify `src/config.py` to tune the optimization dynamics:
- `ITERATIONS`: Optimization epoch count (default: `500`).
- `LEARNING_RATE`: Adam step size (default: `0.03`).
- `WEIGHT_TV`: Total variation penalty weighting (controls pattern smoothness).
- `WEIGHT_NPS`: Non-printability penalty weighting (controls printer gamut adherence).

### 4. Evaluating Physical & Digital Robustness
To run simulation validation or launch the live camera evaluation:
```python
# From within Python or a notebook:
from evaluate import run_digital_validation, run_live_webcam

# 1. Run Monte Carlo transformation simulations
run_digital_validation(perturbation, mask, clean_faces)

# 2. Launch the real-time webcam test with bounding box tracking
run_live_webcam(baseline_img_path="../data/sample/target.jpg")
```

---

## Roadmap
- [x] Whitebox InceptionResnetV1 optimization pipeline.
- [x] Expectation over Transformation (EoT) environmental modeling.
- [x] Sub-pixel boundary smoothing and 300 DPI vector PDF exporter.
- [ ] Black-box transferability pipeline for ArcFace architectures.
- [ ] Multi-subject ensemble optimization.