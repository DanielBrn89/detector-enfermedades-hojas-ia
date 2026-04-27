import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Ruta de tu dataset
DATA_DIR = "data/processed/tomato"

# Transformaciones (IMPORTANTE)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Cargar datasets
train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=val_transforms
)

test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=val_transforms
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Clases
class_names = train_dataset.classes

def get_data():
    return train_loader, val_loader, test_loader, class_names