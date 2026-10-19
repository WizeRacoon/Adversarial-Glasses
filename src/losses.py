import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_dodging_loss(adv_embeddings: torch.Tensor, baseline_embeddings: torch.Tensor) -> torch.Tensor:
    """Dodging: Maximize distance from the true identity."""
    return F.cosine_similarity(adv_embeddings, baseline_embeddings).mean()

def cosine_impersonation_loss(adv_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
    """Impersonation: Minimize distance to the victim's identity."""
    # 1.0 - similarity means as similarity approaches 1 (perfect match), loss approaches 0
    return (1.0 - F.cosine_similarity(adv_embeddings, target_embeddings)).mean()

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, perturbation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Calculate squared differences between adjacent pixels (L2 Norm)
        diff_h = (perturbation[:, :, 1:, :] - perturbation[:, :, :-1, :]) ** 2
        diff_w = (perturbation[:, :, :, 1:] - perturbation[:, :, :, :-1]) ** 2
        
        # Create valid boundary masks (only smooth where BOTH adjacent pixels are inside the frame)
        mask_h = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_w = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        # Sum differences only where the mask is valid
        tv_h = torch.sum(diff_h * mask_h)
        tv_w = torch.sum(diff_w * mask_w)
        
        return tv_h + tv_w

def non_printability_score(perturbation: torch.Tensor, mask: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    colors = colors.view(-1, 3, 1, 1)
    dists = torch.sum((perturbation - colors)**2, dim=1)
    nps_map = torch.prod(dists, dim=0)
    return torch.sum(nps_map * mask[:, 0, :, :])