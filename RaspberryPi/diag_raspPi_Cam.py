import os
import cv2
import socket
import numpy as np
import subprocess

# --- GLOBAL ENVIRONMENT FIX FOR GUI/DSI DISPLAY ---
os.environ["DISPLAY"] = ":0"
os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"
os.environ["QT_QPA_PLATFORM"] = "wayland"

# --- Network Configuration ---
ZEDBOARD_IP = "192.168.10.2"
PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Camera Configuration (Matching your clean 320x240 setup) ---
WIDTH, HEIGHT = 320, 240
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2  # YUV420 format size

rpicam_cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", "30",
    "--codec", "yuv420",
    "-o", "-"
]

print(f"🚀 Starting camera stream to {ZEDBOARD_IP}:{PORT}...")
print("💡 Press 'q' inside a preview window or Ctrl+C in the terminal to exit.")

process = subprocess.Popen(rpicam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

try:
    while True:
        raw_frame = process.stdout.read(FRAME_SIZE)
        if len(raw_frame) != FRAME_SIZE:
            print("⚠️ Warning: Incomplete frame read from camera.")
            break

        # 1. Convert raw YUV420 buffer to BGR for a clean image frame
        yuv = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
        bgr_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        # 2. Convert to Grayscale and resize to 32x32
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

        # 3. INT16 Quantization & Normalization mapping (matching your PyTorch model)
        centered = resized.astype(np.float32) - 127.0
        quantized = np.int16(centered)

        # 4. Send exactly 2048 bytes (32x32 int16) via UDP
        sock.sendto(quantized.tobytes(), (ZEDBOARD_IP, PORT))

        # 5. Display Window 1: Live camera view
        cv2.imshow("Live Camera Feed", bgr_frame)

        # 6. Display Window 2: What the network receives (upscaled 32x32 for visibility)
        preview_stream = cv2.resize(resized, (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Outbound Stream Preview (32x32)", preview_stream)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping stream.")
finally:
    process.terminate()
    process.wait()
    cv2.destroyAllWindows()
    sock.close()
    print("Streamer shut down successfully.")
