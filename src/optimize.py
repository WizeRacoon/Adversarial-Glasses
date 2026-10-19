import os
import argparse
import torch
from torch.optim import Adam
from PIL import Image, ImageOps

import config
from utils import load_glasses_mask, save_smoothed_pdf
from eot import EoTPipeline
from losses import TotalVariationLoss, non_printability_score, cosine_dodging_loss, cosine_impersonation_loss
from models import load_resnet, load_mtcnn

def load_face_batch(data_dir: str, device: torch.device) -> torch.Tensor:
    """Loads and aligns a batch of baseline face images using MTCNN."""
    mtcnn = load_mtcnn(device)
    faces = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    for file in os.listdir(data_dir):
        if file.endswith(valid_extensions):
            path = os.path.join(data_dir, file)
            try:
                img_pil = ImageOps.exif_transpose(Image.open(path).convert('RGB'))
                face_tensor, prob = mtcnn(img_pil, return_prob=True)
                
                if face_tensor is not None and prob > 0.90:
                    real_face = (face_tensor / 255.0).clamp(0, 1)
                    faces.append(real_face)
                else:
                    print(f"Warning: MTCNN failed to confidently detect a face in {file}")
            except Exception as e:
                print(f"Warning: Skipping {file} due to load error: {e}")
                
    if not faces:
        raise ValueError(f"No valid images could be aligned and loaded from {data_dir}!")
        
    return torch.stack(faces).to(device)

def main():
    # --- COMMAND PATTERN ---
    parser = argparse.ArgumentParser(description="Optimize Adversarial Glasses")
    parser.add_argument("--attack", type=str, choices=["dodging", "impersonation"], default="dodging",
                        help="The objective of the adversarial attack.")
    parser.add_argument("--target_dir", type=str, default="",
                        help="Path to target identity faces (required for impersonation).")
    parser.add_argument("--out_dir", type=str, default=config.OUTPUT_DIR,
                        help="Override default output directory for experiment tracking.")
    args = parser.parse_args()

    print(f"Initializing {args.attack.upper()} attack on {config.DEVICE}...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Model & Math
    resnet = load_resnet(config.DEVICE)
    eot = EoTPipeline(config.AFFINE_DEGREES, config.AFFINE_TRANSLATE, config.AFFINE_SCALE)
    tv_loss_fn = TotalVariationLoss()
    
    printable_colors = torch.tensor([
        [0.,0.,0.], [1.,1.,1.], [1.,0.,0.], [0.,1.,0.], 
        [0.,0.,1.], [1.,1.,0.], [0.,1.,1.], [1.,0.,1.]
    ]).to(config.DEVICE)

    # 2. Load Batch Data & Baselines
    clean_faces = load_face_batch(config.DATA_DIR, config.DEVICE)
    with torch.no_grad():
        baseline_embeddings = resnet(clean_faces).mean(dim=0, keepdim=True)
    
    # --- DISPATCH PREPARATION ---
    target_embeddings = None
    if args.attack == "impersonation":
        if not os.path.exists(args.target_dir):
            raise ValueError("Impersonation requires a valid --target_dir containing the victim's faces.")
        target_faces = load_face_batch(args.target_dir, config.DEVICE)
        with torch.no_grad():
            # Average the target's embeddings to create a stable anchor point
            target_embeddings = resnet(target_faces).mean(dim=0, keepdim=True)

    # 3. Initialize Perturbation and Mask
    mask = load_glasses_mask(config.MASK_TEMPLATE_PATH, config.DEVICE)
    perturbation = torch.rand(1, 3, 224, 224, device=config.DEVICE, requires_grad=True)
    optimizer = Adam([perturbation], lr=config.LEARNING_RATE)

    # 4. Optimization Loop
    for i in range(config.ITERATIONS):
        optimizer.zero_grad()
        
        masked_perturbation = perturbation * mask
        adv_faces = torch.clamp((clean_faces * (1.0 - mask)) + masked_perturbation, 0, 1)
        adv_faces_transformed = eot(adv_faces)
        adv_embeddings = resnet(adv_faces_transformed)
        
        # --- LOSS ROUTING DISPATCH ---
        if args.attack == "dodging":
            loss_attack = cosine_dodging_loss(adv_embeddings, baseline_embeddings)
        elif args.attack == "impersonation":
            loss_attack = cosine_impersonation_loss(adv_embeddings, target_embeddings)
            
        loss_tv = tv_loss_fn(perturbation, mask)
        loss_nps = non_printability_score(perturbation, mask, printable_colors)
        
        total_loss = (config.WEIGHT_DODGE * loss_attack) + (config.WEIGHT_TV * loss_tv) + (config.WEIGHT_NPS * loss_nps)
                     
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturbation.clamp_(0, 1)

        if i % 50 == 0:
            print(f"Iter {i:03d} | Total: {total_loss.item():.4f} | Attack: {loss_attack.item():.4f} | TV: {loss_tv.item():.4f}")
        
    print("Optimization complete.")
    pdf_path = os.path.join(args.out_dir, f"{args.attack}_glasses.pdf")
    save_smoothed_pdf(perturbation, mask, pdf_path)

if __name__ == "__main__":
    main()