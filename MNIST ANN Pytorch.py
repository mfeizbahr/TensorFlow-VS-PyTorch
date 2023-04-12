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

#hyperparameter
batch_size = 32

#dataset and data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

#plot images
classes = tuple(str(i) for i in range(10))

def imshow(imgs):
    imgs = imgs / 2 + 0.5  #unnormalize
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)), cmap='gray')
    plt.show()

# one batch of random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

img_grid = torchvision.utils.make_grid(images[0:25], nrow=5)
imshow(img_grid)

print(' '.join(f'{classes[labels[j]]:5s}' for j in range(25)))

#model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) # flatten input images
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training

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
        images = images.view(-1, 28 * 28).to(device) # flatten the image
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
print(f'[{epoch + 1}] train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}')

plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.87, 1.1])
plt.legend(loc='lower right')
plt.show()



print("--- %s seconds ---" % (time.time() - start_time))