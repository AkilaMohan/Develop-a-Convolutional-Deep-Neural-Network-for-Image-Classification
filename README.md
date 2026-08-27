# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

1.Load and Preprocess Data
2.Get the shape of the first image in the training dataset
3.Get the shape of the first image in the test dataset
4.Train the Model
5.Test the Model
6.Predict on a Single Image
7.Display the image

## PROGRAM

### Name: INFANT MARIA STEFANIE .F

### Register Number: 212224230095
```
```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        #Include your code here

    def forward(self, x):
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer
model =
criterion =
optimizer =

# Train the Model
def train_model(model, train_loader, num_epochs=3):

    # write your code here

        
        
        
        print('Name:        ')
        print('Register Number:       ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))

image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


from torchsummary import summary

model = CNNClassifier()

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

print('Name: Varadaram SK')
print('Register Number: 212223040232')
summary(model, input_size=(1, 28, 28))

model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: Varadaram SK')
        print('Register Number: 212223040232')
        print(
            f'Epoch [{epoch+1}/{num_epochs}], '
            f'Loss: {running_loss/len(train_loader):.4f}'
        )


train_model(model, train_loader)


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    print('Name: Varadaram SK')
    print('Register Number: 212223040232')
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    print('Name: Varadaram SK')
    print('Register Number: 212223040232')

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=test_dataset.classes,
        yticklabels=test_dataset.classes
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('Name: Varadaram SK')
    print('Register Number: 212223040232')
    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=test_dataset.classes
        )
    )


test_model(model, test_loader)


import matplotlib.pyplot as plt


def predict_image(model, image_index, dataset):
    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

    class_names = dataset.classes

    print('Name: Varadaram SK')
    print('Register Number: 212223040232')

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(
        f'Actual: {class_names[label]}\n'
        f'Predicted: {class_names[predicted.item()]}'
    )
    plt.axis("off")
    plt.show()

    print(
        f'Actual: {class_names[label]}, '
        f'Predicted: {class_names[predicted.item()]}'
    )


predict_image(model, image_index=80, dataset=test_dataset)

## Confusion Matrix
<img width="1836" height="922" alt="image" src="https://github.com/user-attachments/assets/fa7145c0-f86f-4f59-9709-d1354eb51748" />


## Classification Report

### New Sample Data Prediction
<img width="1245" height="922" alt="image" src="https://github.com/user-attachments/assets/c35d44cf-90cc-4e2b-99c0-a78200dd0bfd" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
