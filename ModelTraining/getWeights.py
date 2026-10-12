import torch
import torch.nn as nn
import numpy as np

class HardwareCNN(nn.Module):
    def __init__(self):
        super(HardwareCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2) # 32 -> 16 -> 8
        
        self.fc1 = nn.Linear(32 * 8 * 8, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 32 * 8 * 8)
        
        x = self.fc1(x)
        return x

# Load your trained model
model = HardwareCNN()
model.load_state_dict(torch.load("ModelTraining/model_weights.pth"))

# 1. Extract fc1 weights and biases
# Shape will be [10, 2048] for weights, [10] for bias
weights = model.fc1.weight.detach().numpy()
biases = model.fc1.bias.detach().numpy()

# 2. Quantize to Fixed-Point Integers
# Multiply by a scaling factor (e.g., 256 for 8 fractional bits)
SCALE_FACTOR = 256.0
weights_quantized = np.round(weights * SCALE_FACTOR).astype(np.int16)
biases_quantized = np.round(biases * SCALE_FACTOR).astype(np.int32)

# 3. Write to a Verilog-friendly hex file
with open("fc1_weights.mem", "w") as f:
    for neuron in range(10):
        for i in range(2048):
            # Write out as 16-bit hex values
            hex_val = format(int(weights_quantized[neuron, i]) & 0xFFFF, '04x')
            f.write(f"{hex_val}\n")

print("Exported 20,480 weights to fc1_weights.mem")