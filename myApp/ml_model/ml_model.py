# ml_model.py
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from django.conf import settings
import os

# Import or define SimpleCNN
# Option 1: If SimpleCNN is defined in another file (e.g., models.py), import it:
# from ..models import SimpleCNN

# Option 2: If you want to define SimpleCNN here:
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the trained model
model_path = os.path.join(settings.BASE_DIR, 'myApp', 'ml_model', 'art_classifier.pth')
model = SimpleCNN(num_classes=20)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True), strict=False)

# Data transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load CSV file for description lookup
csv_path = os.path.join(settings.BASE_DIR, 'myApp', 'ml_model', 'ArtForms.csv')
art_data = pd.read_csv(csv_path)

def predict_art_form(image_path):
    print("Predicting for image:", image_path)  # Debug print
    
    if not os.path.exists(image_path):
        print("File not found:", image_path)  # Debug print
        return "Error", "Image file not found"

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print("Error opening image:", str(e))  # Debug print
        return "Error", f"Unable to open image: {str(e)}"

    # Load and preprocess the image
    image = transform(image).unsqueeze(0)

    # Predict the art form
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Get the corresponding art form and description
    art_form = art_data.iloc[predicted.item(), 1]
    description = art_data.iloc[predicted.item(), 3]
    return art_form, description
