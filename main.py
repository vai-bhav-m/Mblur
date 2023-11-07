import torch
from losses import photometric_error
from models import ResNet, BasicBlock

# Step 1: Generate 5 images using a model
input_image = torch.randn(
    1, 3, 64, 64
)  # Batch size of 1, 3 input channels, 64x64 input size
model = ResNet(
    in_channels=3, out_channels=15, block=BasicBlock, num_blocks=[2, 2, 2, 2]
)
output_image = model(input_image)
print("Output shape:", output_image.shape)
