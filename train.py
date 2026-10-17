import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import SolarDataset

# 1. Configuration / Hyperparameters
DATA_DIR = "data/MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG_DIR = f"{DATA_DIR}/train/train_images"
TRAIN_ANN_FILE = f"{DATA_DIR}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

BATCH_SIZE = 4  # Keep small (e.g., 2 or 4) to avoid Out-Of-Memory errors on high-res images
IMG_SIZE = (512, 512)
EPOCHS = 3
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Using device: {DEVICE}")

    # 2. Initialize Dataset and DataLoader
    print("Loading dataset...")
    train_dataset = SolarDataset(
        image_dir=TRAIN_IMG_DIR,
        ann_file=TRAIN_ANN_FILE,
        img_size=IMG_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    print(f"Total training samples: {len(train_dataset)}")

    # 3. Build U-Net Model with a pretrained ResNet34 encoder
    print("Initializing U-Net model...")
    model = smp.Unet(
        encoder_name="resnet34",       # Pretrained backbone
        encoder_weights="imagenet",    # Use pretrained weights
        in_channels=3,                 # RGB input images
        classes=1,                     # Binary segmentation (filament vs background)
    ).to(DEVICE)

    # 4. Define Loss Function and Optimizer
    # Combination of Dice Loss (great for segmentation overlap) and Binary Cross Entropy
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Simple Training Loop
    print("Starting training loop...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/{EPOCHS}] Completed. Average Loss: {avg_loss:.4f}")

    # Save the trained baseline model weights
    torch.save(model.state_dict(), "unet_solar_baseline.pth")
    print("Training finished! Model saved as 'unet_solar_baseline.pth'.")

if __name__ == "__main__":
    main()