import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Create a 12x12x3 image numpy array
image_np = np.array([
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  200,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   255,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0, 0, 0], [  0, 0, 0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   100,   253], [  0, 100, 254], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0]],
    [[  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [  0,   0,   0], [0,   0,   0]]
])

# Convert the numpy array to a torch tensor and permute dimensions to (C, H, W)
image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 12, 12)

# Apply max pooling with a kernel size of 2x2 and stride of 2
pooled_tensor = F.avg_pool2d(image_tensor, kernel_size=2, stride=2)

# Permute dimensions back to (H, W, C) and convert the result back to a numpy array
pooled_image_np = pooled_tensor.squeeze(0).permute(1, 2, 0).numpy()

# Plot the original and pooled images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot original image
axs[0].imshow(image_np)
axs[0].set_title("Original Image")
axs[0].axis("off")
print(image_np.shape)

# Plot pooled image
axs[1].imshow(pooled_image_np)
axs[1].set_title("Pooled Image")
axs[1].axis("off")
print(pooled_image_np.shape)

plt.show()