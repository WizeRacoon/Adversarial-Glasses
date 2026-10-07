# Physical Evaluation & Experiments

The `src/evaluate.py` script provides a dedicated Command-Line Interface (CLI) to evaluate the robustness of generated adversarial glasses. It supports live webcam tracking, transferability testing, and automated data collection for academic reporting.

## Prerequisites: Establishing the Baseline
Before running any live tests, the system needs a baseline embedding of your face to compare against.
1. Take a clear, well-lit, frontal headshot of the subject **without** glasses.
2. Save the image to your data directory (e.g., `data/sample/target.jpg` or `data/private/target.jpg`).

## CLI Usage Modes

Navigate to the `src` directory to run the evaluation tool:
```bash
cd src
```

### 1. White-Box Live Validation (`live`)
Tests physical evasion strictly against the InceptionResnetV1 model you optimized against. It provides real-time bounding box color feedback (Green = Evaded, Red = Recognized).

```bash
python evaluate.py --mode live --baseline ../data/sample/target.jpg
```

### 2. Black-Box Transferability Test (`dual`)
Evaluates transferability by running two FRS architectures simultaneously on the live feed. It tracks the ResNet similarity score alongside the ArcFace similarity score to determine if the attack transfers to an unseen feature space.

```bash
python evaluate.py --mode dual --baseline ../data/sample/target.jpg
```

### 3. Quantitative Robustness Profiling (`profile`)
An automated data collection sequence for generating academic plots. This mode records similarity scores alongside bounding box size (distance from camera) and X-axis offset (panning angle) across three distinct physical phases: *No Glasses*, *Black Glasses*, and *Adversarial Glasses*.

```bash
python evaluate.py --mode profile --baseline ../data/sample/target.jpg --frames 100
```
**Experiment Workflow:**
1. The script will prompt you to step into the frame without glasses. Press `SPACE` to begin recording.
2. It will capture the specified number of frames (default: 100), recording distance, angle, and similarity metrics.
3. It will automatically prompt you to swap to the next phase (Black Glasses, then Adversarial).
4. **Outputs:** Upon completion, the script exports a raw `.csv` dataset and quadratic curve trend plots to the `outputs/` directory.

### Digital Simulation Validation (Notebook Integration)
If you wish to run Monte Carlo simulations (Expectation over Transformation) without physical printing, you can import the validation module directly into your Jupyter notebooks or custom scripts:

```python
from evaluate import run_digital_validation

# Simulates camera sensor noise, affine tilts, and color jitter
run_digital_validation(perturbation, mask, clean_faces, iterations=500, threshold=0.3)
```