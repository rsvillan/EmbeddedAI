import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import socket
import struct
import cv2
import numpy as np

# 1. CNN for Color Images (3 channels, 32x32 input based on SVHN dataset)
class ColorCNN(nn.Module):
    def __init__(self):
        super(ColorCNN, self).__init__()
        # 3 input channels (RGB) instead of 1 (Grayscale)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32x32 -> pool(16x16) -> pool(8x8). 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_svhn_model():
    print("Downloading SVHN (Color Digits) Dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # SVHN is real-world color digits from Google Street View
    train_data = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = ColorCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training Color CNN (1 Epoch for speed)...")
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        if batch_idx >= 400: # Cap at 400 for quick testing
            break
            
    print("Training Complete!")
    return model

def main():
    model = train_svhn_model()
    model.eval()

    # Setup TCP Server
    HOST = '0.0.0.0'
    PORT = 8080
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"\nListening for 1080p TCP stream on port {PORT}...")
    conn, addr = server_socket.accept()
    print(f"Connected to Raspberry Pi at {addr}")

    data = b""
    payload_size = struct.calcsize("Q") # 8-byte standard size for unpacking data lengths

    with torch.no_grad():
        while True:
            # 1. Receive the size of the incoming JPEG frame
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet
            if not data: break

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            # 2. Receive the actual JPEG frame bytes
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 3. Decode JPEG to 1080p full-color frame
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame_1080p = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame_1080p is not None:
                # 4. Show the raw 1080p stream
                cv2.imshow("1080p Stream from Pi", frame_1080p)

                # 5. Preprocess for Neural Network (Resize to 32x32, convert to PyTorch Tensor)
                # (In a real app, you would crop the center where the number is, then resize)
                resized_for_ml = cv2.resize(frame_1080p, (32, 32))
                tensor_img = transforms.ToTensor()(resized_for_ml).unsqueeze(0)
                tensor_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor_img)

                # 6. Inference
                outputs = model(tensor_img)
                
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted].item()

                if (confidence > .8):
                    print(f"Predicted Digit: {predicted}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()