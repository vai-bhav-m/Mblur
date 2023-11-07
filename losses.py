import torch
import torch.nn.functional as F


def photometric_error(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the photometric_error (using MSE loss) between 2 input images
    """
    mse_loss = F.mse_loss(img1, img2)
    return mse_loss
