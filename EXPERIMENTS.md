# Physical Evaluation & Experiments

The `src/evaluate.py` script provides a dedicated Command-Line Interface (CLI) to evaluate the robustness of generated adversarial glasses. It supports live webcam tracking, transferability testing, and automated data collection for academic reporting.

## Prerequisites: Establishing the Baseline
Before running any live tests, the system needs a baseline embedding of your face to compare against.
1. Take a clear, well-lit, frontal headshot of the subject **without** glasses. Ideally, this image is as close as possible to the average image during the experiment.
2. Save the image to your data directory (e.g., `data/sample/target.jpg` or `data/private/target.jpg`).

## Automated Experiment Pipelines (Recommended)

The easiest way to run a full end-to-end experiment (optimization + evaluation + plotting) is using the unified bash runners. These scripts automatically create timestamped artifact directories in `outputs/`.

```bash
# Run from the root of the repository
./experiments/1_optimize_dodging.sh
# Physically Print generated eyeglasses pattern
./experiments/2_optimize_dodging.sh
```
* **Phase 1:** Optimizes the adversarial glasses against the batched faces in your data directory.
* **Phase 2:** Automatically launches the live robustness profiler and saves the resulting CSV and trend plots into the timestamped experiment folder.

---

## Manual CLI Usage Modes

If you prefer to run evaluation modules individually, navigate to the `src` directory:
```bash
cd src
```

### 1. White-Box Live Validation (`live`)
Tests physical evasion strictly against the InceptionResnetV1 model you optimized against. It provides real-time bounding box color feedback (Green = Evaded, Red = Recognized).

```bash
python evaluate.py --mode live --baseline ../data/sample/target.jpg
```

### 2. Black-Box Transferability Test (`dual`)
Evaluates transferability by running two FRS architectures simultaneously on the live feed. It tracks the ResNet similarity score alongside the ArcFace similarity score in real-time to determine if the attack transfers to an unseen feature space.

```bash
python evaluate.py --mode dual --baseline ../data/sample/target.jpg
```

### 3. Quantitative Dual-Architecture Profiling (`profile`)
An automated data collection sequence for generating academic plots. This mode records similarities for **both** ResNet and ArcFace simultaneously alongside bounding box size (distance) and X-axis offset (angle) across three distinct physical phases: *No Glasses*, *Black Glasses*, and *Adversarial Glasses*.

```bash
python evaluate.py --mode profile --baseline ../data/sample/target.jpg --out_dir ../outputs
```
**Interactive UI Controls:**
When the profiling window opens, you can use the following keys before recording each phase:
* **`+` / `=`**: Increase the number of frames to record.
* **`-`**: Decrease the number of frames (useful for quick functionality tests).
* **`SPACE`**: Start recording the current phase.
* **`q`**: Abort the evaluation early (safely exports any data collected up to that point).

The script automatically exports `dual_robustness.csv`, `resnet_robustness_boundaries.png`, and `arcface_robustness_boundaries.png` to your output directory.

### Digital Simulation Validation (Notebook Integration)
If you wish to run Monte Carlo simulations (Expectation over Transformation) without physical printing, you can import the validation module directly into your Jupyter notebooks or custom scripts:

```python
from evaluate import run_digital_validation

# Simulates camera sensor noise, affine tilts, and color jitter
run_digital_validation(perturbation, mask, clean_faces, iterations=500, threshold=0.3)
```