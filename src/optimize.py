import os
import torch
from torch.optim import Adam
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize
from torchvision.transforms.functional import resize

import config
from utils import load_glasses_mask, save_smoothed_pdf
from eot import EoTPipeline
from losses import TotalVariationLoss, non_printability_score, cosine_dodging_loss
from models import load_resnet

def load_face_batch(data_dir: str, device: torch.device) -> torch.Tensor:
    """Loads a batch of baseline face images."""
    faces = []
    # Added upper/lowercase variations and .jpeg
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    for file in os.listdir(data_dir):
        if file.endswith(valid_extensions):
            path = os.path.join(data_dir, file)
            try:
                # PIL safely opens the file and forces it to 3-channel RGB
                img_pil = Image.open(path).convert('RGB')
                
                # to_tensor automatically converts to float and scales to [0.0, 1.0]
                img_tensor = to_tensor(img_pil) 
                img_resized = resize(img_tensor, (224, 224))
                faces.append(img_resized)
            except Exception as e:
                print(f"Warning: Skipping {file} due to load error: {e}")
                
    if not faces:
        raise ValueError(f"No valid images could be loaded from {data_dir}!")
        
    return torch.stack(faces).to(device)

def main():
    print(f"Initializing on {config.DEVICE}...")
    
    # 1. Load Model & Math
    resnet = load_resnet(config.DEVICE)
    eot = EoTPipeline(config.AFFINE_DEGREES, config.AFFINE_TRANSLATE, config.AFFINE_SCALE)
    tv_loss_fn = TotalVariationLoss()
    
    # Placeholder printable colors (pure RGB + CMYK corners for demonstration)
    printable_colors = torch.tensor([
        [0.,0.,0.], [1.,1.,1.], [1.,0.,0.], [0.,1.,0.], 
        [0.,0.,1.], [1.,1.,0.], [0.,1.,1.], [1.,0.,1.]
    ]).to(config.DEVICE)

    # 2. Load Batch Data & Baselines
    clean_faces = load_face_batch(config.DATA_DIR, config.DEVICE)
    with torch.no_grad():
        baseline_embeddings = resnet(clean_faces)
    print(f"Loaded {len(clean_faces)} baseline faces.")

    # 3. Initialize Perturbation and Mask
    mask = load_glasses_mask(config.MASK_TEMPLATE_PATH, config.DEVICE)
    perturbation = torch.rand(1, 3, 224, 224, device=config.DEVICE, requires_grad=True)
    optimizer = Adam([perturbation], lr=config.LEARNING_RATE)

    # 4. Optimization Loop
    for i in range(config.ITERATIONS):
        optimizer.zero_grad()
        
        # Mask the perturbation so only the frame pixels are optimized
        masked_perturbation = perturbation * mask
        
        # Overlay the masked glasses onto the clean faces (replacing the pixels underneath)
        adv_faces = (clean_faces * (1.0 - mask)) + masked_perturbation
        adv_faces = torch.clamp(adv_faces, 0, 1)
        
        # Apply EoT
        adv_faces_transformed = eot(adv_faces)
        
        # Forward Pass
        adv_embeddings = resnet(adv_faces_transformed)
        
        # Calculate Joint Loss (Ensure TV and NPS only penalize the frame pixels)
        loss_dodge = cosine_dodging_loss(adv_embeddings, baseline_embeddings)
        loss_tv = tv_loss_fn(perturbation, mask)
        loss_nps = non_printability_score(perturbation, mask, printable_colors)
        
        total_loss = (config.WEIGHT_DODGE * loss_dodge) + \
                     (config.WEIGHT_TV * loss_tv) + \
                     (config.WEIGHT_NPS * loss_nps)
                     
        total_loss.backward()
        optimizer.step()

        # Force the raw pixels back into the 0.0 to 1.0 RGB range
        with torch.no_grad():
            perturbation.clamp_(0, 1)

        if i % 50 == 0:
            print(f"Iter {i:03d} | Total: {total_loss.item():.4f} | "
                  f"Dodge: {loss_dodge.item():.4f} | TV: {loss_tv.item():.4f}")
        
    print("Optimization complete. Ready for physical export.")

    # 5. Export Physical PNG
    pdf_path = os.path.join(config.OUTPUT_DIR, "physical_adversarial_glasses.pdf")
    save_smoothed_pdf(perturbation, mask, pdf_path)

if __name__ == "__main__":
    main()