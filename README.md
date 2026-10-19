# Adversarial Eyeglass Framework

This repository serves as the codebase for an MSc research project focusing on physical adversarial evasion attacks against state-of-the-art Face Recognition Systems (FRSs). Building upon the foundational research of Sharif et al. in *"Accessorize to a Crime"*, this project reproduces, modernizes, and extends the physical realization of adversarial eyeglass frames. 

The repository provides a complete, differentiable pipeline for optimizing printable adversarial textures. It actively integrates Expectation over Transformation (EoT), Total Variation (TV) constraints, and a Non-Printability Score (NPS) to bridge the digital-to-physical performance gap.

**Current Scope & Completed Objectives**
The primary focus of this internship was establishing a highly robust framework for **dodging** (untargeted evasion). The pipeline maximizes real-world physical robustness through:
- **Subject-Specific Robustness:** Implemented multi-image batching across diverse lighting conditions and poses to prevent spatial overfitting and ensure reliable evasion in the physical domain.

**Future Research Directions**
The codebase utilizes a modular Command & Dispatch architecture, leaving a structured foundation for future students and subsequent internships to explore:
- **Ensemble-Based Optimization:** Integrating the black-box transferability module directly into the training loop via a PyTorch-native ArcFace implementation to generate universal perturbations capable of simultaneous cross-architecture evasion.
- **Targeted Impersonation Attacks:** (Initial loss routing provided).
- **Hardware Fluctuations and Morphological Impact:** (e.g., varying frame surface areas).
- **Defensive Countermeasures and Mitigation.**

## Repository Structure

```text
Adversarial-Glasses/
├── data/
│   ├── sample/                   # Public/stock face images for baseline testing
│   ├── private/                  # Personal evaluation faces (IGNORED by Git)
│   └── glasses_template.png      # High-contrast RGBA eyeglass silhouette
├── experiments/                  # Unified bash runners for standardized testing
│   ├── run_dodging.sh            
│   └── run_batch_dodging.sh            
├── notebooks/                    # Prototyping and custom exploratory analyses
├── outputs/                      # Generated PDFs, evaluation plots, and textures
├── src/
│   ├── config.py                 # Pipeline hyperparameters and constants
│   ├── eot.py                    # Expectation over Transformation augmentations
│   ├── evaluate.py               # Digital simulation and live webcam testing
│   ├── losses.py                 # Total Variation, NPS, and Cosine Dodging losses
│   ├── models.py                 # Model wrappers (InceptionResnetV1, MTCNN, ArcFace)
│   ├── optimize.py               # Optimization loop and batch driver
│   ├── plotting.py               # Decoupled matplotlib visualization routines
│   └── utils.py                  # Masking routines, inpainting, and PDF rendering
├── .gitignore
├── EXPERIMENTS.md                # Documentation for running physical tests
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
- Multiple images in these directories will be automatically batched for spatial robustness during optimization.
- Keep the transparent mask file at `data/glasses_template.png`.
- *Note:* The repository's `.gitignore` rules prevent any media inside `data/private/` or stray image files from being tracked by Git.

### 2. Generating Adversarial Glasses
To run the optimization pipeline manually:
```bash
cd src
python optimize.py --attack dodging --data_dir ../data/private/my_dataset
```
- The optimizer calculates joint loss (Cosine Dodging, Total Variation for smoothness, and Non-Printability Score for color realizability) across all batched faces simultaneously.
- When optimization finishes, a high-resolution, vector-scaled PDF with cutting contours is exported to the outputs directory.

### 3. Hyperparameter Configuration
Modify `src/config.py` to tune the optimization dynamics:
- `ITERATIONS`: Optimization epoch count (default: `500`).
- `LEARNING_RATE`: Adam step size (default: `0.03`).
- `WEIGHT_TV`: Total variation penalty weighting (controls pattern smoothness).
- `WEIGHT_NPS`: Non-printability penalty weighting (controls printer gamut adherence).

### 4. Evaluation and Profiling
The framework includes a comprehensive CLI tool for testing digital robustness and conducting physical live-camera experiments against white-box and black-box models. 

**Please refer to [EXPERIMENTS.md](EXPERIMENTS.md) for detailed instructions on running unified bash experiments and live webcam tests.**

---

## Roadmap / Accomplishments
- [x] Whitebox InceptionResnetV1 optimization pipeline.
- [x] Expectation over Transformation (EoT) environmental modeling.
- [x] Sub-pixel boundary smoothing and 300 DPI vector PDF exporter.
- [x] Black-box transferability pipeline for ArcFace architectures.
- [x] Subject-Specific Robustness via Multi-Image Batching optimization.
- [ ] *Future Work:* Targeted Impersonation Attacks (transitioning from general dodging to targeting a specific identity's embedding).
- [ ] *Future Work:* Ensemble-Based Optimization (PyTorch-native ArcFace integration for joint-objective cross-architecture transferability).
- [ ] *Future Work:* Hardware Fluctuations and Morphological Impact (varying camera sensors, ISPs, capture angles, and frame surface areas).
- [ ] *Future Work:* Deconstructing the Digital-to-Physical Delta (analyzing printer gamut clipping, ambient color temperature, and hardware smoothing).
- [ ] *Future Work:* Defensive Countermeasures and Mitigation (anomaly detection and adversarial training).