import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 1. Define the Hardware-Accurate CNN
class HardwareCNN(nn.Module):
    def __init__(self):
        super(HardwareCNN, self).__init__()
        # Layer 1: 3x3 Convolution (1 channel in, 1 channel out, no bias)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        
        # Layer 2: Max Pooling (2x2 kernel, stride 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Dense/Fully Connected (16x16 input features -> 10 classes, no bias)
        self.fc = nn.Linear(16 * 16, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)  # Hardware ReLU
        x = self.pool(x)   # Hardware MaxPool (32x32 -> 16x16)
        x = x.view(-1, 16 * 16) # Flatten for CPU
        x = self.fc(x)
        return x

def main():
    # 2. Load CIFAR-10 and Convert to Grayscale
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = HardwareCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop (Updated for 10 Epochs)
    print("Training hardware-accurate model...")
    model.train()
    
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199:
                print(f"Epoch {epoch+1} | Batch {i+1} | Loss: {running_loss / 200:.3f}")
                running_loss = 0.0

    # 4. Extract, Quantize, and Export Weights
    print("\nExporting INT16 weights for ZedBoard...")
    
    # We multiply by 256 (or your chosen scaling factor) to convert floats to INT16
    quantization_scale = 256.0 

    # Export Conv1 Weights (9 total weights)
    conv1_weights = model.conv1.weight.data.numpy().flatten()
    conv1_int16 = np.int16(np.round(conv1_weights * quantization_scale))
    with open("conv1_weight.bin", "wb") as f:
        f.write(conv1_int16.tobytes())
    print(f"Saved conv1_weight.bin: {conv1_int16.shape} array")

    # Export FC Weights (10 neurons * 256 inputs = 2560 total weights)
    fc_weights = model.fc.weight.data.numpy()
    fc_int16 = np.int16(np.round(fc_weights * quantization_scale))
    with open("fc_weight.bin", "wb") as f:
        f.write(fc_int16.tobytes())
    print(f"Saved fc_weight.bin: {fc_int16.shape} matrix")

if __name__ == "__main__":
    main()