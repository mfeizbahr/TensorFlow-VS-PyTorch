import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters
batch_size = 32
input_size = 32 * 32 * 3
hidden_size = 128
output_size = 10
learning_rate = 0.001

# Load dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define model
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, input_size) # Flatten the input images to a vector
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ANN(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
start_time = time.time()
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
steps_per_epoch = len(trainloader)

for epoch in range(10):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for (inputs, labels) in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward + loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / steps_per_epoch
    train_acc = correct / total
    history['loss'].append(train_loss)
    history['accuracy'].append(train_acc)

    # Evaluation
running_loss = 0.0
correct = 0
total = 0
model.train()

for (inputs, labels) in trainloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    #forward + loss
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    #backward + optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

train_loss = running_loss / steps_per_epoch
train_acc = correct / total
history['loss'].append(train_loss)
history['accuracy'].append(train_acc)

#evaluation
test_loss = 0.0
test_correct = 0
test_total = 0
model.eval()

with torch.no_grad():
    for (images, labels) in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        #take class with highest value as validation.

        _, predicted = torch.max(outputs.data, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_loss += criterion(outputs, labels).item()

val_loss = test_loss / len(testloader)
val_acc = test_correct / test_total
history['val_loss'].append(val_loss)
history['val_accuracy'].append(val_acc)

