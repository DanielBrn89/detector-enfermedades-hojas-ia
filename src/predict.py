import torch
from torchvision import transforms
from PIL import Image
from torchvision import models

MODEL_PATH = "models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Transformación igual que entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_percent = confidence.item() * 100

    return predicted_class, confidence_percent