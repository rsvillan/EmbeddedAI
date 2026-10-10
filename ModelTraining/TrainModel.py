import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class HardwareFriendlyCNN(nn.Module):
    def __init__(self):
        super(HardwareFriendlyCNN, self).__init__()
        
        # Input: 32x32x3 RGB image
        # Conv1: 3x3 kernel, 3 channels in, 16 channels out, padding=1 to keep spatial size 32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        # MaxPool1: 2x2 window. Reduces spatial size from 32x32 to 16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2: 3x3 kernel, 16 channels in, 32 channels out, padding=1 to keep spatial size 16x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        # MaxPool2: 2x2 window. Reduces spatial size from 16x16 to 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected (Dense) Layer
        # After pool2, we have 32 channels of 8x8 feature maps (32 * 8 * 8 = 2048 parameters)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # Flatten for the dense layer
        x = self.fc(x)
        return x

def main():
    # using GPU
    device = torch.device("cuda")

    # Load and Normalize CIFAR-10 Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map to [-1, 1] range
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Initialize Model, Loss Function, and Optimizer
    model = HardwareFriendlyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    epochs = 10
    print("Starting FP32 Training...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(trainloader):.4f} - Accuracy: {epoch_acc:.2f}%")

    print("Finished Training.")

    # 6. Save the FP32 Model
    torch.save(model.state_dict(), "cifar10_fp32_model.pt")
    print("Model saved to cifar10_fp32_model.pt")

# Start
if __name__ == '__main__':
    main()