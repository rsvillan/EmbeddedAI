import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# 1. Define the strictly linear model (No biases, matching FPGA BRAM limits)
class FPGAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10, bias=False) 

    def forward(self, x):
        x = x.view(-1, 784) # Flatten the 28x28 image
        return self.fc(x)

# 2. Download and load MNIST dataset
print("Downloading MNIST dataset...")
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Setup training
model = FPGAModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Quick Training Loop (1 Epoch)
print("Training model (1 Epoch)...")
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 200 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

print("Training complete!")

# 5. Extract, Quantize, and Export
weights = model.fc.weight.detach().cpu().numpy()

# Quantize float32 weights to INT8 (-128 to 127)
max_val = np.max(np.abs(weights))
scale = 127.0 / max_val
int8_weights = np.round(weights * scale).astype(np.int8)

# Export to raw binary file
int8_weights.tofile("weights.bin")
print(f"Successfully exported {int8_weights.size} bytes to weights.bin")