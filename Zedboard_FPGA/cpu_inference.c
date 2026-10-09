// Standard libraries for input/output, memory management, and data types
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

// Linux networking libraries for UDP sockets
#include <sys/socket.h>
#include <arpa/inet.h>

#define PORT 8080 // The UDP port we will listen on

// ==============================================================================
// GLOBAL WEIGHT ARRAYS
// We declare these globally (in the BSS segment) rather than inside main().
// If we put 421KB of arrays inside main(), it could cause a "Stack Overflow" 
// crash on the ZedBoard's embedded Linux OS.
// Array dimensions match PyTorch: [Output Channels][Input Channels][Height][Width]
// ==============================================================================
int8_t conv1_w[32][1][3][3];    // Conv1 Weights: 32 filters, 1 input channel, 3x3 kernel
int8_t conv1_b[32];             // Conv1 Biases: 32 values
int8_t conv2_w[64][32][3][3];   // Conv2 Weights: 64 filters, 32 input channels, 3x3 kernel
int8_t conv2_b[64];             // Conv2 Biases: 64 values
int8_t fc1_w[128][3136];        // FC1 Weights: 128 output nodes, 3136 input nodes (64*7*7)
int8_t fc1_b[128];              // FC1 Biases: 128 values
int8_t fc2_w[10][128];          // FC2 Weights: 10 output classes (digits 0-9), 128 input nodes
int8_t fc2_b[10];               // FC2 Biases: 10 values

// Function to load the binary weights from the SD card / filesystem into memory
int load_weights() {
    FILE *f = fopen("weights.bin", "rb"); // Open file in Read-Binary mode
    if (!f) return -1; // Return error if file doesn't exist
    
    // Read the exact number of bytes for each layer in the exact order 
    // that the Python script exported them.
    fread(conv1_w, 1, 288, f);     // 32 * 1 * 3 * 3 = 288 bytes
    fread(conv1_b, 1, 32, f);      // 32 bytes
    fread(conv2_w, 1, 18432, f);   // 64 * 32 * 3 * 3 = 18,432 bytes
    fread(conv2_b, 1, 64, f);      // 64 bytes
    fread(fc1_w, 1, 401408, f);    // 128 * 3136 = 401,408 bytes
    fread(fc1_b, 1, 128, f);       // 128 bytes
    fread(fc2_w, 1, 1280, f);      // 10 * 128 = 1,280 bytes
    fread(fc2_b, 1, 10, f);        // 10 bytes
    
    fclose(f); // Close the file to free OS resources
    return 0;
}

int main() {
    // 1. Load the weights into memory at startup
    if (load_weights() < 0) {
        printf("ERROR: weights.bin not found! Ensure the 421KB file is in this directory.\n");
        return -1;
    }
    printf("Loaded 421KB int8 SimpleCNN weights into CPU memory.\n");

    // 2. Set up the UDP network socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0); // Create a UDP socket
    struct sockaddr_in server_addr, client_addr; 
    uint8_t pixel_buffer[784];                   // Buffer to hold incoming 28x28 image

    server_addr.sin_family = AF_INET;            // IPv4
    server_addr.sin_addr.s_addr = INADDR_ANY;    // Listen on all network interfaces (Ethernet, etc)
    server_addr.sin_port = htons(PORT);          // Set the port (8080)
    
    // Bind the socket to the port so the OS knows to route packets here
    bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr));
    
    printf("ZedBoard SimpleCNN active. Waiting for UDP stream on port %d...\n", PORT);

    // 3. Infinite loop to process incoming video frames forever
    while (1) {
        socklen_t client_len = sizeof(client_addr);
        
        // Wait here until a packet arrives. 'bytes_received' will hold the packet size.
        int bytes_received = recvfrom(sockfd, pixel_buffer, sizeof(pixel_buffer), 0, 
                                      (struct sockaddr *)&client_addr, &client_len);
        
        // Only process the frame if it is exactly 784 bytes (a complete 28x28 image)
        if (bytes_received == 784) {
            
            // --- ALLOCATE MEMORY FOR LAYER OUTPUTS ---
            uint8_t input_img[28][28];             // 2D array for the raw image
            int32_t conv1_out[32][28][28] = {0};   // Conv1 output: 32 channels of 28x28
            int32_t pool1_out[32][14][14] = {0};   // Pool1 output: 32 channels of 14x14 (downsampled)
            
            // Note: We use 64-bit integers (int64_t) from here on. Because we are multiplying 
            // unscaled integers, the numbers get huge. 64-bit prevents the math from overflowing.
            int64_t conv2_out[64][14][14] = {0};   // Conv2 output: 64 channels of 14x14
            int64_t pool2_out[64][7][7] = {0};     // Pool2 output: 64 channels of 7x7 (downsampled)
            int64_t fc1_out[128] = {0};            // FC1 output: 128 flat nodes
            int64_t fc2_out[10] = {0};             // FC2 output: 10 flat nodes (final scores)

            // Step 0: Convert the flat 1D 784-byte UDP buffer into a 2D 28x28 grid
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    input_img[y][x] = pixel_buffer[y * 28 + x];
                }
            }

            // Step 1: Convolution 1 (1 channel -> 32 channels) + ReLU Activation
            for (int k = 0; k < 32; k++) {                  // Loop through all 32 output filters
                for (int y = 0; y < 28; y++) {              // Loop through image height
                    for (int x = 0; x < 28; x++) {          // Loop through image width
                        int32_t sum = conv1_b[k];           // Start math with the bias value
                        
                        // Slide the 3x3 kernel window over the pixels
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int ny = y + dy, nx = x + dx; // Calculate neighbor pixel coordinates
                                
                                // Padding Check: Only multiply if the neighbor is actually inside the 28x28 image
                                if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
                                    // Multiply pixel by weight and accumulate
                                    sum += (int32_t)input_img[ny][nx] * conv1_w[k][0][dy+1][dx+1];
                                }
                            }
                        }
                        // ReLU function: If the sum is positive, keep it. If negative, set to 0.
                        conv1_out[k][y][x] = (sum > 0) ? sum : 0; 
                    }
                }
            }

            // Step 2: Max Pooling 1 (Shrinks 28x28 down to 14x14)
            for (int k = 0; k < 32; k++) {                  // Loop through all 32 channels
                for (int y = 0; y < 14; y++) {              // Note: Loop only goes to 14 now
                    for (int x = 0; x < 14; x++) {
                        // Look at a 2x2 grid of pixels from the previous layer
                        int32_t max_val = conv1_out[k][y*2][x*2]; // Top-left
                        // Compare with top-right, bottom-left, bottom-right. Keep the largest value.
                        if (conv1_out[k][y*2][x*2+1] > max_val) max_val = conv1_out[k][y*2][x*2+1];
                        if (conv1_out[k][y*2+1][x*2] > max_val) max_val = conv1_out[k][y*2+1][x*2];
                        if (conv1_out[k][y*2+1][x*2+1] > max_val) max_val = conv1_out[k][y*2+1][x*2+1];
                        
                        // Save the maximum value to the pooled output
                        pool1_out[k][y][x] = max_val;
                    }
                }
            }

            // Step 3: Convolution 2 (32 channels -> 64 channels) + ReLU Activation
            for (int k_out = 0; k_out < 64; k_out++) {              // Loop through 64 new filters
                for (int y = 0; y < 14; y++) {                      // Height is 14
                    for (int x = 0; x < 14; x++) {                  // Width is 14
                        int64_t sum = conv2_b[k_out];               // Start with bias (64-bit int)
                        
                        for (int k_in = 0; k_in < 32; k_in++) {     // Must sum across all 32 input channels!
                            // Slide 3x3 kernel
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    int ny = y + dy, nx = x + dx;
                                    // Padding check
                                    if (ny >= 0 && ny < 14 && nx >= 0 && nx < 14) {
                                        sum += (int64_t)pool1_out[k_in][ny][nx] * conv2_w[k_out][k_in][dy+1][dx+1];
                                    }
                                }
                            }
                        }
                        // ReLU Activation
                        conv2_out[k_out][y][x] = (sum > 0) ? sum : 0; 
                    }
                }
            }

            // Step 4: Max Pooling 2 (Shrinks 14x14 down to 7x7)
            for (int k = 0; k < 64; k++) {                  // Loop through 64 channels
                for (int y = 0; y < 7; y++) {               // Loop to 7
                    for (int x = 0; x < 7; x++) {
                        // Look at 2x2 grid and find the maximum value
                        int64_t max_val = conv2_out[k][y*2][x*2];
                        if (conv2_out[k][y*2][x*2+1] > max_val) max_val = conv2_out[k][y*2][x*2+1];
                        if (conv2_out[k][y*2+1][x*2] > max_val) max_val = conv2_out[k][y*2+1][x*2];
                        if (conv2_out[k][y*2+1][x*2+1] > max_val) max_val = conv2_out[k][y*2+1][x*2+1];
                        
                        pool2_out[k][y][x] = max_val;
                    }
                }
            }

            // Step 5: Fully Connected Layer 1 (Flatten 64*7*7 into 128 nodes) + ReLU
            for (int out = 0; out < 128; out++) {           // Loop through 128 output nodes
                int64_t sum = fc1_b[out];                   // Start with bias
                int in_idx = 0;                             // Tracks flat index for the weights
                
                // Iterating through the 3D pool array flattens it automatically
                for (int k = 0; k < 64; k++) {
                    for (int y = 0; y < 7; y++) {
                        for (int x = 0; x < 7; x++) {
                            // Multiply pixel by weight, then increment weight index
                            sum += pool2_out[k][y][x] * fc1_w[out][in_idx++];
                        }
                    }
                }
                // ReLU Activation
                fc1_out[out] = (sum > 0) ? sum : 0; 
            }

            // Step 6: Fully Connected Layer 2 (128 nodes to 10 final classes)
            for (int out = 0; out < 10; out++) {            // Loop through 10 digit classes (0-9)
                int64_t sum = fc2_b[out];
                for (int in = 0; in < 128; in++) {          // Multiply against the 128 nodes from FC1
                    sum += fc1_out[in] * fc2_w[out][in];
                }
                fc2_out[out] = sum;                         // No ReLU on the final output!
            }

            // Step 7: Argmax Classification (Find the digit with the highest score)
            // Initialize max score to the lowest possible number
            int64_t max_score = -999999999999999LL; 
            int predicted_digit = -1;
            
            for (int i = 0; i < 10; i++) {
                if (fc2_out[i] > max_score) {
                    max_score = fc2_out[i];   // Update highest score
                    predicted_digit = i;      // Save the winning digit index
                }
            }

            // Print the final result!
            printf("CNN Predicted Digit: %d (Raw Score: %lld)\n", predicted_digit, (long long)max_score);
        }
    }
    
    // Close network socket when done (though this infinite loop never naturally exits)
    close(sockfd);
    return 0;
}