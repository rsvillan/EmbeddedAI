import torch
import torch.nn as nn
import numpy as np
import os

# Re-define the architecture to load the weights
class HardwareFriendlyCNN(nn.Module):
    def __init__(self):
        super(HardwareFriendlyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

def quantize_tensor(tensor, num_bits=16):
    """
    Quantizes a floating-point tensor to fixed-point integer.
    For INT16, the range is -32768 to 32767.
    """
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    
    # Find the maximum absolute value in the tensor to determine the scale
    max_val = torch.max(torch.abs(tensor)).item()
    
    # If the max value is 0, avoid division by zero
    if max_val == 0:
        scale = 1.0
    else:
        scale = qmax / max_val
        
    # Apply scale, round to nearest integer, and clamp to limits
    q_tensor = torch.round(tensor * scale)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    
    return q_tensor.detach().numpy().astype(np.int16), scale

def export_to_binary(q_numpy_array, filename):
    """Flattens the NumPy array and writes it as raw binary."""
    with open(filename, "wb") as f:
        f.write(q_numpy_array.flatten().tobytes())
    print(f"Exported {filename} | Shape: {q_numpy_array.shape} | Size: {q_numpy_array.nbytes} bytes")

def main():
    # Load the trained FP32 model
    model = HardwareFriendlyCNN()
    model.load_state_dict(torch.load("ModelTraining/cifar10_fp32_model.pt", map_location=torch.device('cpu')))
    model.eval()

    output_dir = "quantized_weights"
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Starting INT16 Quantization & Export ---")
    
    # Iterate through the named parameters (weights and biases)
    scale_factors = {}
    for name, param in model.named_parameters():
        layer_name = name.replace('.', '_')
        
        # Quantize the tensor
        q_tensor, scale = quantize_tensor(param)
        scale_factors[layer_name] = scale
        
        # Export to binary
        filename = os.path.join(output_dir, f"{layer_name}.bin")
        export_to_binary(q_tensor, filename)
        
    print("\n--- Scale Factors (Needed for Activation Scaling in C/Verilog) ---")
    for name, scale in scale_factors.items():
        print(f"{name}: {scale:.4f}")

if __name__ == '__main__':
    main()