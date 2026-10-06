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