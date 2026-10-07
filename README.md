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
│   ├── models.py                 # Model wrappers (InceptionResnetV1, MTCNN, ArcFace)
│   ├── optimize.py               # Optimization loop and batch driver
│   └── utils.py                  # Masking routines, inpainting, and PDF rendering
├── .gitignore
├── EXPERIMENTS.md                # Documentation for running physical tests
├── README.md
└── requirements.txt