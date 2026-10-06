import torch
import torch.nn as nn

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

def cosine_dodging_loss(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    # We want to maximize distance, so we maximize cosine similarity (push it toward -1)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(embeddings1, embeddings2).mean()