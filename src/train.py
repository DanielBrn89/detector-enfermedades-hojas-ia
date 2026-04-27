import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.data_loader import get_data

# Configuración
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = "models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

train_loader, val_loader, test_loader, class_names = get_data()
num_classes = len(class_names)

# Modelo ResNet18 preentrenado
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Congelar capas base
for param in model.parameters():
    param.requires_grad = False

# Cambiar última capa para tus clases
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nÉpoca {epoch + 1}/{EPOCHS}")

    # Entrenamiento
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # Validación
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Loss entrenamiento: {train_loss:.4f}")
    print(f"Accuracy entrenamiento: {train_acc:.4f}")
    print(f"Accuracy validación: {val_acc:.4f}")

    # Guardar mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names
        }, MODEL_PATH)

        print("✅ Mejor modelo guardado")

print("\nEntrenamiento finalizado")
print("Mejor accuracy validación:", best_val_acc)
print("Modelo guardado en:", MODEL_PATH)