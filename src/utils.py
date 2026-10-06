import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F

def load_glasses_mask(template_path: str, device: torch.device) -> torch.Tensor:
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Mask template not found at {template_path}")
        
    img_pil = Image.open(template_path).convert("RGBA")
    tensor = F.to_tensor(img_pil)
    alpha = tensor[3:4, :, :] 
    
    # Scale proportionally to fit within 224x224
    _, h, w = alpha.shape
    scale = 224 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    alpha_resized = F.resize(alpha, (new_h, new_w))
    
    # Pad to 224x224 to center the glasses on the canvas
    pad_left = (224 - new_w) // 2
    pad_top = (224 - new_h) // 2
    pad_right = 224 - new_w - pad_left
    pad_bottom = 224 - new_h - pad_top
    
    mask = F.pad(alpha_resized, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
    
    # Binarize
    mask = (mask > 0.5).float()
    return mask.unsqueeze(0).to(device)

def save_smoothed_pdf(perturbation: torch.Tensor, mask: torch.Tensor, output_path: str):
    """Upscales, smooths, and saves the adversarial pattern as a printable PDF."""
    # 1. Extract raw pattern and binary mask from GPU
    final_pattern = np.ascontiguousarray(
        (perturbation * mask).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    )
    mask_2d = np.ascontiguousarray(mask.squeeze(0)[0].detach().cpu().numpy())

    # 2. Crop to active bounding box
    active_pixels = np.argwhere(mask_2d > 0.5)
    y_min, x_min = active_pixels.min(axis=0)
    y_max, x_max = active_pixels.max(axis=0)
    
    cropped_pattern = final_pattern[y_min:y_max, x_min:x_max]
    cropped_mask = mask_2d[y_min:y_max, x_min:x_max]

    # 3. High-Resolution Upsampling (8x Super-Sampling)
    scale_factor = 8
    hr_h = cropped_pattern.shape[0] * scale_factor
    hr_w = cropped_pattern.shape[1] * scale_factor

    # 4. Color Inpainting to prevent black edge bleeding
    mask_uint8 = (cropped_mask > 0.5).astype(np.uint8) * 255
    pattern_uint8 = (cropped_pattern * 255).astype(np.uint8)
    inpainted_pattern = cv2.inpaint(pattern_uint8, 255 - mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA) / 255.0

    hr_pattern = cv2.resize(inpainted_pattern, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    hr_pattern = np.clip(hr_pattern, 0.0, 1.0)

    # 5. Smoothstep Anti-Aliasing on Mask Boundary
    hr_binary = cv2.resize((cropped_mask > 0.5).astype(np.float32), (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
    ksize = int(scale_factor * 4) | 1  
    sigma = scale_factor * 1.5         
    blurred_mask = cv2.GaussianBlur(hr_binary, (ksize, ksize), sigma)
    hr_mask = np.clip((blurred_mask - 0.48) / 0.04, 0.0, 1.0)[..., None]

    # 6. Alpha blend onto white paper background
    white_bg = np.ones_like(hr_pattern)
    printable_glasses = hr_pattern * hr_mask + white_bg * (1.0 - hr_mask)

    # 7. Draw faint gray scissor cutting guide
    contour_mask = (hr_mask.squeeze() > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(printable_glasses, contours, -1, (0.75, 0.75, 0.75), thickness=2, lineType=cv2.LINE_AA)

    # 8. Physical Dimensions (14.5 cm width)
    target_width_cm = 14.50
    width_inch = target_width_cm / 2.54
    aspect_ratio = cropped_pattern.shape[0] / cropped_pattern.shape[1]
    height_inch = width_inch * aspect_ratio

    # 9. Render PDF at 300 DPI
    fig = plt.figure(figsize=(width_inch, height_inch), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(printable_glasses)

    plt.savefig(output_path, format='pdf', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"Physical export saved to: {output_path}")