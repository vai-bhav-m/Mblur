import torch
from losses import photometric_error
from models import UNet, PoseNetCNN

# Step 1: Generate 5 images using a model
model = UNet()
input_image = torch.randn(16, 3, 64, 64)  
output_image = model(input_image)
print("Output shape after passing through UNet:", output_image.shape) 

# Step 2: imposing supervision loss on first image
selected_channels = [0, 1, 2]
Loss_supervision = photometric_error(input_image, output_image[:,selected_channels,:,:])
print(f"Supervision loss: {Loss_supervision}    Type: {type(Loss_supervision)}")

# Passing pairs of images to the PoseNet to generate poses
pose_net = PoseNetCNN()
poses = []
for i in range(3,13,3):
    base_img = output_image[:,selected_channels,:,:]
    warp_img = output_image[:,i:i+3,:,:]
    concat_img = torch.cat((base_img, warp_img), dim=1)
    poses.append(pose_net(concat_img))
print("Poses:", len(poses))

# Warp Images



