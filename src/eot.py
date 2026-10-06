import torch
import torchvision.transforms as T

class EoTPipeline:
    def __init__(self, degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)):
        self.transform = T.Compose([
            T.RandomAffine(degrees=degrees, translate=translate, scale=scale),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies random transformations to a batch of face tensors.
        Expects x of shape (B, C, H, W) normalized between [0, 1].
        """
        return self.transform(x)