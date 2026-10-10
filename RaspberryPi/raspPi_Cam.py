import cv2
import socket
import numpy as np
import subprocess
import os

# Create a copy of the current environment variables to allow DSI output over SSH
env = os.environ.copy()
env["DISPLAY"] = ":0"
env["XDG_RUNTIME_DIR"] = "/run/user/1000"

# --- Network Configuration ---
ZEDBOARD_IP = "192.168.10.2"
PORT = 8080

# UDP socket for low-overhead transmission
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Camera Configuration ---
width, height = 800, 480
bytes_per_frame = int(width * height * 1.5) # Total YUV420 frame size
luma_size = width * height                  # Size of just the grayscale data

# Hardware-render the feed directly to the DSI display
cmd = [
    "rpicam-vid",
    "--width", str(width),
    "--height", str(height),
    "--framerate", "30",
    "--codec", "yuv420",
    "--timeout", "0",
    "--fullscreen",
    "--output", "-"
]

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env
)

if process.poll() is not None:
    stdout, stderr = process.communicate()
    print(f"rpicam-vid failed with error:\n{stderr.decode('utf-8')}")
    exit(1)

print(f"Hardware preview rendering to DSI display...")
print(f"Streaming 32x32 INT16 payload over Ethernet to {ZEDBOARD_IP}:{PORT}...")

try:
    while True:
        # 1. Grab the raw YUV frame
        raw_data = process.stdout.read(bytes_per_frame)
        if len(raw_data) != bytes_per_frame:
            continue

        # 2. CPU OPTIMIZATION: Extract ONLY the Grayscale (Y) channel.
        luma_data = raw_data[:luma_size]
        gray = np.frombuffer(luma_data, dtype=np.uint8).reshape((height, width))

        # 3. Center Crop (square out of the rectangular frame)
        start_x = (width // 2) - (height // 2)
        cropped = gray[0:height, start_x:start_x+height]

        # 4. Resize to 32x32 for the FPGA Line Buffers
        resized = cv2.resize(cropped, (32, 32), interpolation=cv2.INTER_AREA)

        # 5. INT16 Quantization & Scaling
        # Multiply by 40 to push the 0-255 range up, preventing Verilog truncation from destroying data
        quantized = np.int16(resized) * 40

        # 6. Send exactly 2048 bytes (1024 16-bit words) to the ZedBoard via UDP
        sock.sendto(quantized.tobytes(), (ZEDBOARD_IP, PORT))

except KeyboardInterrupt:
    print("\nShutting down pipeline.")

finally:
    process.terminate()
    sock.close()