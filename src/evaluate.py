import os
import cv2
import torch
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize
import torch.nn.functional as F

import plotting
import config
from eot import EoTPipeline
from models import load_resnet, load_mtcnn, load_arcface
from utils import load_glasses_mask

def run_digital_validation(perturbation: torch.Tensor, mask: torch.Tensor, 
                           clean_faces: torch.Tensor, iterations: int = 500, 
                           threshold: float = 0.3):
    resnet = load_resnet(config.DEVICE)
    eot = EoTPipeline(config.AFFINE_DEGREES, config.AFFINE_TRANSLATE, config.AFFINE_SCALE)
    
    with torch.no_grad():
        baseline_embeddings = resnet(clean_faces)
        adv_faces = (clean_faces * (1.0 - mask)) + (perturbation * mask)
        adv_faces = torch.clamp(adv_faces, 0, 1)

        sim_scores = []
        for _ in range(iterations):
            distorted = eot(adv_faces)
            embeddings = resnet(distorted)
            sim = F.cosine_similarity(embeddings, baseline_embeddings).mean().item()
            sim_scores.append(sim)

    sim_array = np.array(sim_scores)
    mean_sim = np.mean(sim_array)
    success_rate = np.mean(sim_array < threshold) * 100.0

    # Save summary histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sim_scores, bins=35, color='royalblue', edgecolor='black', alpha=0.75)
    ax.axvline(threshold, color='crimson', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax.axvline(mean_sim, color='black', linestyle='-', linewidth=2, label=f'Mean ({mean_sim:.2f})')
    ax.set_title(f'Digital Robustness Simulation (Success Rate: {success_rate:.1f}%)')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(config.OUTPUT_DIR, "robustness_distribution.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"\n--- Digital Robustness Validation ({iterations} runs) ---")
    print(f"Mean Similarity: {mean_sim:.4f}")
    print(f"Standard Deviation: {np.std(sim_array):.4f}")
    print(f"Evasion Success Rate: {success_rate:.1f}%")
    print(f"Distribution plot saved to: {plot_path}")

def run_live_webcam(baseline_img_path: str, threshold: float = 0.3):
    device = config.DEVICE
    mtcnn = load_mtcnn(device)
    resnet = load_resnet(device)

    # Compute reference embedding
    if not os.path.exists(baseline_img_path):
        raise FileNotFoundError(f"Baseline image not found at: {baseline_img_path}")
        
    pil_base = Image.open(baseline_img_path).convert('RGB')
    face_tensor, _ = mtcnn(pil_base, return_prob=True)
    if face_tensor is None:
        raise RuntimeError("Could not detect face in baseline image.")
        
    base_tensor = (face_tensor.unsqueeze(0).to(device) / 255.0).clamp(0, 1)
    with torch.no_grad():
        original_embedding = resnet(base_tensor)

    cap = cv2.VideoCapture(0)
    print("\nStarting live webcam test. Press 'q' in the window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        face_crop, prob = mtcnn(pil_frame, return_prob=True)
        boxes, _ = mtcnn.detect(pil_frame)

        if face_crop is not None and prob > 0.90 and boxes is not None:
            live_tensor = (face_crop.unsqueeze(0).to(device) / 255.0).clamp(0, 1)
            with torch.no_grad():
                live_embedding = resnet(live_tensor)
                similarity = F.cosine_similarity(live_embedding, original_embedding).item()

            evaded = similarity < threshold
            color = (0, 255, 0) if evaded else (0, 0, 255)
            status = "DODGED" if evaded else "RECOGNIZED"

            x1, y1, x2, y2 = [int(coord) for coord in boxes[0]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{status} | Sim: {similarity:.4f}", (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Searching for face...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Physical Evasion Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_baselines(baseline_img_path: str, device: torch.device):
    """Utility to extract baseline embeddings for both architectures."""
    mtcnn = load_mtcnn(device)
    resnet = load_resnet(device)
    app = load_arcface(device)
    
    pil_img = Image.open(baseline_img_path).convert('RGB')
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # ResNet Baseline
    face_tensor, _ = mtcnn(pil_img, return_prob=True)
    if face_tensor is None:
        raise ValueError("MTCNN failed to detect face in baseline image.")
    real_face = (face_tensor.unsqueeze(0).to(device) / 255.0).clamp(0, 1)
    with torch.no_grad():
        resnet_baseline = resnet(real_face)
        
    # ArcFace Baseline
    arc_faces = app.get(cv_img)
    if not arc_faces:
        raise ValueError("ArcFace failed to detect face in baseline image.")
    raw_emb = arc_faces[0].embedding
    arc_baseline = raw_emb / np.linalg.norm(raw_emb)
    
    return resnet_baseline, arc_baseline, mtcnn, resnet, app


def run_dual_live_webcam(baseline_img_path: str, resnet_threshold: float = 0.3, arcface_threshold: float = 0.4):
    """Live webcam feed tracking evasion across ResNet (White-Box) and ArcFace (Black-Box)."""
    device = config.DEVICE
    resnet_baseline, arc_baseline, mtcnn, resnet, app = get_baselines(baseline_img_path, device)
    
    cap = cv2.VideoCapture(0)
    print("\nStarting Dual-Architecture Live Feed. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # 1. ResNet Pipeline
            resnet_sim = 0.0
            face_tensor, prob = mtcnn(pil_img, return_prob=True)
            if face_tensor is not None and prob > 0.90:
                live_face = (face_tensor.unsqueeze(0).to(device) / 255.0).clamp(0, 1)
                with torch.no_grad():
                    live_embedding = resnet(live_face)
                    resnet_sim = F.cosine_similarity(live_embedding, resnet_baseline).item()
                    
            # 2. ArcFace Pipeline
            arc_sim = 0.0
            arc_faces = app.get(frame)
            if arc_faces:
                live_arc_emb = arc_faces[0].embedding
                live_arc_emb = live_arc_emb / np.linalg.norm(live_arc_emb)
                arc_sim = float(np.dot(live_arc_emb, arc_baseline))
                
                # Draw ArcFace bounding box
                bbox = arc_faces[0].bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)
                
            # 3. UI Display
            cv2.putText(frame, f"ResNet Sim: {resnet_sim:.4f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"ArcFace Sim: {arc_sim:.4f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            cv2.imshow('Black-Box Transferability Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_physical_profiling(baseline_img_path: str, frames_per_phase: int = 100, out_dir: str = None):
    """Records face coordinates and dual-model similarities across physical phases."""
    device = config.DEVICE
    out_dir = out_dir or config.OUTPUT_DIR
    resnet_baseline, arc_baseline, mtcnn, resnet, app = get_baselines(baseline_img_path, device)
    
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_center_x = frame_width / 2

    phases = ['No Glasses', 'Black Glasses', 'Adversarial Glasses']
    recorded_data = {phase: [] for phase in phases}

    current_target_frames = frames_per_phase

    print("Starting Dual-Architecture Profiling. Press 'q' in the window at any time to abort.")
    
    abort = False
    try:
        for phase in phases:
            if abort: break
                
            print(f"\n--- Phase: {phase} ---")
            recording = False
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                arc_sim, res_sim, face_size_px, offset_x = None, None, None, None
                
                # 1. ArcFace Evaluation
                arc_faces = app.get(frame)
                if arc_faces:
                    bbox = arc_faces[0].bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    face_size_px = x2 - x1  
                    face_center_x = (x1 + x2) / 2
                    offset_x = face_center_x - frame_center_x  
                    
                    live_arc_emb = arc_faces[0].embedding
                    live_arc_emb = live_arc_emb / np.linalg.norm(live_arc_emb)
                    arc_sim = float(np.dot(live_arc_emb, arc_baseline))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    
                # 2. ResNet Evaluation
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                face_tensor, prob = mtcnn(pil_img, return_prob=True)
                
                if face_tensor is not None and prob > 0.90:
                    live_face = (face_tensor.unsqueeze(0).to(device) / 255.0).clamp(0, 1)
                    with torch.no_grad():
                        live_embedding = resnet(live_face)
                        res_sim = F.cosine_similarity(live_embedding, resnet_baseline).item()

                if recording:
                    if arc_sim is not None and res_sim is not None:
                        frame_count += 1
                        recorded_data[phase].append((frame_count, face_size_px, offset_x, arc_sim, res_sim))
                        
                        cv2.putText(frame, f"RECORDING {phase}... {frame_count}/{current_target_frames}", 
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"ResNet: {res_sim:.2f} | Arc: {arc_sim:.2f}", 
                                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        
                        if frame_count >= current_target_frames:
                            break
                    else:
                        cv2.putText(frame, "Waiting for face detection...", (30, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Pre-recording UI with Live Adjustments
                    cv2.putText(frame, f"Put on: {phase}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Frames: {current_target_frames} (Press +/- to adjust)", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE to Record", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.putText(frame, "Press 'q' to Quit", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                cv2.imshow('Dual-Architecture Profiling', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    recording = True
                    print(f"Recording {phase} started! Target: {current_target_frames} frames.")
                elif key == ord('q'):
                    print("Profiling interrupted by user. Exporting collected data...")
                    abort = True
                    break
                elif not recording:
                    # Adjust frames live
                    if key == ord('=') or key == ord('+'):
                        current_target_frames += 10
                    elif key == ord('-') and current_target_frames > 10:
                        current_target_frames -= 10
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # --- Data Export ---
    csv_path = os.path.join(out_dir, "dual_robustness.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Phase', 'Frame', 'Face_Width_px', 'Center_Offset_px', 'ArcFace_Similarity', 'ResNet_Similarity'])
        for phase in phases:
            for row in recorded_data[phase]:
                writer.writerow([phase] + list(row))

    # --- Plotting Helper ---
    def plot_model_trends(model_name, threshold, sim_idx, output_filename):
        colors = {'No Glasses': 'blue', 'Black Glasses': 'orange', 'Adversarial Glasses': 'red'}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Physical Robustness Boundaries ({model_name})', fontsize=16)

        for phase in phases:
            if recorded_data[phase]:
                data = list(zip(*recorded_data[phase]))
                sizes = np.array(data[1])
                offsets = np.array(data[2])
                sims = np.array(data[sim_idx])  # Index 3 for ArcFace, 4 for ResNet
                
                ax1.scatter(sizes, sims, alpha=0.25, color=colors[phase], edgecolors='none')
                ax2.scatter(offsets, sims, alpha=0.25, color=colors[phase], edgecolors='none')
                
                idx_sizes = np.argsort(sizes)
                p_size = np.poly1d(np.polyfit(sizes, sims, 2))
                ax1.plot(sizes[idx_sizes], p_size(sizes[idx_sizes]), color=colors[phase], linewidth=2, label=phase)
                
                idx_offsets = np.argsort(offsets)
                p_offset = np.poly1d(np.polyfit(offsets, sims, 2))
                ax2.plot(offsets[idx_offsets], p_offset(offsets[idx_offsets]), color=colors[phase], linewidth=2, label=phase)

        ax1.axhline(y=threshold, color='k', linestyle='--', label='Detection Threshold')
        ax1.set_xlabel('Face Bounding Box Width (Pixels)')
        ax1.set_ylabel(f'Cosine Similarity ({model_name})')
        ax1.legend()

        ax2.axhline(y=threshold, color='k', linestyle='--', label='Detection Threshold')
        ax2.set_xlabel('Distance from Frame Center (Pixels)')
        ax2.legend()

        plot_path = os.path.join(out_dir, output_filename)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        return plot_path

    # Generate both plots
    arc_path = plot_model_trends("ArcFace", 0.4, 3, "arcface_robustness_boundaries.png")
    res_path = plot_model_trends("ResNet", 0.3, 4, "resnet_robustness_boundaries.png")
    
    print(f"Data saved to: {csv_path}")
    print(f"Plots saved to: {arc_path} and {res_path}")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Physical Evaluation & Profiling Suite")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["live", "dual", "profile"], 
                        help="Evaluation mode to run.")
    parser.add_argument("--baseline", type=str, default="../data/sample/target.jpg", 
                        help="Path to your baseline reference face image.")
    parser.add_argument("--frames", type=int, default=100, 
                        help="Number of frames to capture per phase (for profiling).")
    parser.add_argument("--out_dir", type=str, default="../outputs", 
                        help="Output directory for experiment data.")

    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        print(f"Error: Baseline image not found at '{args.baseline}'.")
        exit(1)

    if args.mode == "live":
        run_live_webcam(args.baseline)
    elif args.mode == "dual":
        run_dual_live_webcam(args.baseline)
    elif args.mode == "profile":
        run_physical_profiling(args.baseline, frames_per_phase=args.frames, out_dir=args.out_dir)