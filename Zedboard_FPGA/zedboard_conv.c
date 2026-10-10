#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h> 

#define DMA_BASE_ADDR   0x40400000 
#define IP_BASE_ADDR    0x40000000 
#define SRC_PHYS_ADDR   0x1E000000 
#define DST_PHYS_ADDR   0x1E100000

#define MM2S_DMACR      (0x00 / 4)
#define MM2S_DMASR      (0x04 / 4)
#define MM2S_SA         (0x18 / 4)
#define MM2S_LENGTH     (0x28 / 4)

#define S2MM_DMACR      (0x30 / 4)
#define S2MM_DMASR      (0x34 / 4)
#define S2MM_DA         (0x48 / 4)
#define S2MM_LENGTH     (0x58 / 4)

#define TRANSFER_WORDS  1024
#define TRANSFER_BYTES  (TRANSFER_WORDS * sizeof(uint16_t)) 
#define UDP_PORT        8080

const char* cifar10_classes[] = {
    "Airplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
};

int main(void) {
    // 1. Memory Mapping for DMA and Custom IP
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("Failed to open /dev/mem");
        return 1;
    }

    volatile uint32_t *dma = (volatile uint32_t *)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DMA_BASE_ADDR);
    volatile uint32_t *ip_ctrl = (volatile uint32_t *)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IP_BASE_ADDR);
    volatile uint16_t *src = (volatile uint16_t *)mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd, SRC_PHYS_ADDR);
    volatile uint16_t *dst = (volatile uint16_t *)mmap(NULL, 8192, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DST_PHYS_ADDR);

    // 2. Load PyTorch Convolutional Weights (Hardware)
    printf("Loading PyTorch weights (conv1_weight.bin)...\n");
    FILE *weight_file = fopen("conv1_weight.bin", "rb");
    if (weight_file) {
        int16_t pytorch_weights[9];
        fread(pytorch_weights, sizeof(int16_t), 9, weight_file);
        fclose(weight_file);
        for (int i = 0; i < 9; i++) { ip_ctrl[i] = pytorch_weights[i]; }
    } else {
        printf("Warning: conv1_weight.bin not found. Using empty weights.\n");
    }

    // 3. Load PyTorch Fully Connected Weights (Software/CPU)
    printf("Loading FC weights (fc_weight.bin) to CPU memory...\n");
    int16_t cpu_fc_weights[10][TRANSFER_WORDS] = {0};
    FILE *fc_file = fopen("fc_weight.bin", "rb");
    if (fc_file) {
        fread(cpu_fc_weights, sizeof(int16_t), 10 * TRANSFER_WORDS, fc_file);
        fclose(fc_file);
    } else {
        printf("Warning: fc_weight.bin not found. CPU will output zeros.\n");
    }

    // 4. Setup UDP Socket
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(UDP_PORT);
    
    bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr));
    printf("Listening for UDP frames on port %d...\n", UDP_PORT);

    // 5. Main Inference Loop Variables
    uint16_t recv_buffer[TRANSFER_WORDS];
    int frame_count = 0;
    
    struct timeval last_print_time, current_time;
    gettimeofday(&last_print_time, NULL); 

    while (1) {
        // Wait for a 2048-byte payload from the Raspberry Pi
        int n = recvfrom(sockfd, recv_buffer, TRANSFER_BYTES, 0, (struct sockaddr *)&client_addr, &client_len);
        if (n == TRANSFER_BYTES) {
            frame_count++;
            
            // Copy network payload directly to DMA source memory
            memcpy((void*)src, recv_buffer, TRANSFER_BYTES);

            // Reset and Trigger DMA
            dma[MM2S_DMACR] = 0x0004; dma[S2MM_DMACR] = 0x0004; usleep(100);
            
            // S2MM (Receiver): The Max Pooler reduced the 32x32 (1024) image to 16x16 (256)
            dma[S2MM_DMACR] = 0x0001; 
            dma[S2MM_DA] = DST_PHYS_ADDR; 
            dma[S2MM_LENGTH] = (TRANSFER_BYTES / 4) * 2; 
            
            // MM2S (Transmitter): Still sending the full 32x32 image
            dma[MM2S_DMACR] = 0x0001; 
            dma[MM2S_SA] = SRC_PHYS_ADDR; 
            dma[MM2S_LENGTH] = TRANSFER_BYTES * 2; 

            // Poll for completion
            int timeout = 100000;
            while (!(dma[MM2S_DMASR] & 0x0002) && --timeout > 0);
            while (!(dma[S2MM_DMASR] & 0x0002) && --timeout > 0);

            // --- ARM CPU: Fully Connected Layer ---
            int32_t class_scores[10] = {0};
            
            for (int neuron = 0; neuron < 10; neuron++) {
                // Loop updated to 256 (TRANSFER_WORDS / 4) to match the new 16x16 Max Pooled feature map
                for (int pixel = 0; pixel < (TRANSFER_WORDS / 4); pixel++) {
                    class_scores[neuron] += (int16_t)dst[pixel] * cpu_fc_weights[neuron][pixel];
                }
            }

            // --- Classification Argmax ---
            int best_class = 0;
            int32_t max_score = -2147483648; 
            
            for (int i = 0; i < 10; i++) {
                if (class_scores[i] > max_score) {
                    max_score = class_scores[i];
                    best_class = i;
                }
            }

            // --- Time-Gated Print Logic ---
            gettimeofday(&current_time, NULL);
            double elapsed_seconds = (current_time.tv_sec - last_print_time.tv_sec) + 
                                     (current_time.tv_usec - last_print_time.tv_usec) / 1000000.0;
                                     
            if (elapsed_seconds >= 0.5) {
                printf("Processed Frame %d | Classification: %s\n", frame_count, cifar10_classes[best_class]);
                last_print_time = current_time; 
            }
        }
    }

    munmap((void *)dma, 4096); 
    munmap((void *)ip_ctrl, 4096);
    munmap((void *)src, 8192); 
    munmap((void *)dst, 8192);
    close(fd);
    return 0;
}