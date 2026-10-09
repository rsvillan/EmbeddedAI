import cv2
import numpy as np

# --- Configuration ---
PI_IP = "192.168.86.83" # REPLACE WITH PI'S IP ADDRESS
PORT = 8080

# Connect to the Pi's TCP video stream using OpenCV's built-in network reader
stream_url = f"tcp://{PI_IP}:{PORT}"
print(f"Connecting to Pi stream at {stream_url}...")

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Failed to open stream. Make sure the Pi script is running first.")
    exit()

print("Receiving 1080p video...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or dropped.")
            break

        # 1. Show the full, beautiful 1080p feed from the Pi
        # Resize slightly just so it fits on your laptop screen comfortably
        display_frame = cv2.resize(frame, (1280, 720))
        
        # --- Create the MNIST 28x28 Preview ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_dim = min(h, w)
        start_x = (w // 2) - (min_dim // 2)
        start_y = (h // 2) - (min_dim // 2)
        
        # Crop the center square and shrink to 28x28
        cropped = gray[start_y:start_y+min_dim, start_x:start_x+min_dim]
        resized_28 = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
        inverted_28 = cv2.bitwise_not(resized_28)
        
        # Blow the 28x28 image back up to 280x280 so you can see exactly what the FPGA will see
        fpga_preview = cv2.resize(inverted_28, (280, 280), interpolation=cv2.INTER_NEAREST)

        # Draw a box on the 1080p frame showing where it's cropping for the FPGA
        cv2.rectangle(display_frame, 
                     (start_x*1280//1920, start_y*720//1080), 
                     ((start_x+min_dim)*1280//1920, (start_y+min_dim)*720//1080), 
                     (0, 255, 0), 2)

        # Show both windows side-by-side
        cv2.imshow("Full 1080p Camera View", display_frame)
        cv2.imshow("What the FPGA Sees (28x28)", fpga_preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    cap.release()
    cv2.destroyAllWindows()