import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define dataset class
class ArtDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Create a mapping from string labels to numeric indices
        self.label_mapping = {label: idx for idx, label in enumerate(self.data['ArtForm'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 2]  # Path of the image
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx, 1]  # ArtForm
        
        # Convert the label to a numeric index using the mapping
        label = self.label_mapping[label]

        if self.transform:
            image = self.transform(image)

        # Ensure label is a tensor of type Long for CrossEntropyLoss
        label = torch.tensor(label).long()  
        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Use the full path for the CSV file
csv_file_path = os.path.join(os.getcwd(), 'ArtForms.csv')  # Adjust this if necessary
dataset = ArtDataset(csv_file=csv_file_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training the model
model = SimpleCNN(num_classes=len(dataset.label_mapping))  # Use the number of unique classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to the appropriate device (if using GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(10):  # Number of epochs
    for images, labels in dataloader:
        # Move images and labels to the device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'art_classifier.pth')
