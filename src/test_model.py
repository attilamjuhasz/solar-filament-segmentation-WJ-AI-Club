import torch

from model import SolarUNet


# Create model
model = SolarUNet(num_classes=4)

print("Model created!")


# Create a fake 512x512 grayscale image
x = torch.randn(1, 1, 512, 512)

print("Input shape:", x.shape)


# Run through model
with torch.no_grad():
    output = model(x)


print("Output shape:", output.shape)
