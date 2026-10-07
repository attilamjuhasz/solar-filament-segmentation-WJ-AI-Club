from dataset import SolarFilamentDataset


dataset = SolarFilamentDataset(
    "data/MAGFiLO_1.0_Kaggle_2026"
)


print("Dataset size:")
print(len(dataset))


# Get first image
sample = dataset[0]


print("\n===== IMAGE =====")

print("Shape:", sample["image"].shape)
print("Data type:", sample["image"].dtype)


print("\n===== MASKS =====")

print("Shape:", sample["masks"].shape)
print("Data type:", sample["masks"].dtype)


print("\n===== CATEGORIES =====")

print(sample["categories"])


print("\n===== FILE =====")

print(sample["filename"])