#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/mman.h>

#define BIND_IP             "0.0.0.0"
#define PORT                8080

#define IMG_PIXELS          784
#define IN_DMA_BYTES        (784 * 2)           
#define OUT_DMA_WORDS       (26 * 26 * 8) 
#define OUT_DMA_BYTES       (OUT_DMA_WORDS * 2) 

#define FIXED_POINT_SHIFT   8

#define AXI_DMA_BASE        0x40400000    
#define DMA_REG_SPAN        0x10000
#define TX_BUF_PADDR        0x1E000000
#define RX_BUF_PADDR        0x1E100000
#define BUF_SPAN            0x100000      

#define MM2S_DMACR          0x00
#define MM2S_DMASR          0x04
#define MM2S_SA             0x18
#define MM2S_LENGTH         0x28

#define S2MM_DMACR          0x30
#define S2MM_DMASR          0x34
#define S2MM_DA             0x48
#define S2MM_LENGTH         0x58

static int16_t FC1_WEIGHTS[32][1352];
static int16_t FC1_BIAS[32];
static int16_t FC2_WEIGHTS[10][32];
static int16_t FC2_BIAS[10];

static inline void dma_set(volatile uint32_t *dma, uint32_t offset, uint32_t val) { dma[offset >> 2] = val; }
static inline uint32_t dma_get(volatile uint32_t *dma, uint32_t offset) { return dma[offset >> 2]; }

void print_dma_status(volatile uint32_t *dma) {
    uint32_t mm2s = dma_get(dma, MM2S_DMASR);
    uint32_t s2mm = dma_get(dma, S2MM_DMASR);
    printf("  [DMA] MM2S: 0x%08X | S2MM: 0x%08X\n", mm2s, s2mm);
}

void dma_reset(volatile uint32_t *dma) {
    dma_set(dma, MM2S_DMACR, 0x4);
    dma_set(dma, S2MM_DMACR, 0x4);
    while (dma_get(dma, MM2S_DMACR) & 0x4);
    while (dma_get(dma, S2MM_DMACR) & 0x4);
}

int load_fc_weights(const char *dir) {
    char path[256]; FILE *f;
    snprintf(path, sizeof(path), "%s/fc1_w_int16.bin", dir);
    if (!(f = fopen(path, "rb"))) return -1; fread(FC1_WEIGHTS, 2, 32 * 1352, f); fclose(f);
    snprintf(path, sizeof(path), "%s/fc1_b_int16.bin", dir);
    if (!(f = fopen(path, "rb"))) return -1; fread(FC1_BIAS, 2, 32, f); fclose(f);
    snprintf(path, sizeof(path), "%s/fc2_w_int16.bin", dir);
    if (!(f = fopen(path, "rb"))) return -1; fread(FC2_WEIGHTS, 2, 10 * 32, f); fclose(f);
    snprintf(path, sizeof(path), "%s/fc2_b_int16.bin", dir);
    if (!(f = fopen(path, "rb"))) return -1; fread(FC2_BIAS, 2, 10, f); fclose(f);
    return 0;
}

int run_arm_classifier(const int16_t *conv_features) {
    static int16_t relu_conv[8][26][26];
    int idx = 0;
    for (int r = 0; r < 26; r++) {
        for (int c = 0; c < 26; c++) {
            for (int f = 0; f < 8; f++) {
                int16_t val = conv_features[idx++];
                relu_conv[f][r][c] = (val > 0) ? val : 0;
            }
        }
    }

    static int16_t pool_out[1352];
    int p_idx = 0;
    for (int f = 0; f < 8; f++) {
        for (int r = 0; r < 26; r += 2) {
            for (int c = 0; c < 26; c += 2) {
                int16_t m = relu_conv[f][r][c];
                if (relu_conv[f][r+1][c] > m) m = relu_conv[f][r+1][c];
                if (relu_conv[f][r][c+1] > m) m = relu_conv[f][r][c+1];
                if (relu_conv[f][r+1][c+1] > m) m = relu_conv[f][r+1][c+1];
                pool_out[p_idx++] = m;
            }
        }
    }

    static int32_t fc1_out[32];
    for (int i = 0; i < 32; i++) {
        int64_t acc = 0;
        for (int j = 0; j < 1352; j++) {
            acc += (int32_t)pool_out[j] * FC1_WEIGHTS[i][j];
        }
        acc = (acc >> FIXED_POINT_SHIFT) + FC1_BIAS[i];
        fc1_out[i] = (acc > 0) ? (int32_t)(acc >> FIXED_POINT_SHIFT) : 0;
    }

    int32_t best_logit = -2147483647;
    int predicted_digit = -1;
    
    printf("Raw Logits: ");
    for (int i = 0; i < 10; i++) {
        int64_t acc = 0;
        for (int j = 0; j < 32; j++) acc += fc1_out[j] * FC2_WEIGHTS[i][j];
        acc = (acc >> FIXED_POINT_SHIFT) + FC2_BIAS[i];
        
        printf("[%d]:%d ", i, (int)acc);
        
        if (acc > best_logit) { best_logit = acc; predicted_digit = i; }
    }
    printf("\n");
    return predicted_digit;
}

int main(void) {
    int dev_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (dev_mem < 0) return EXIT_FAILURE;

    volatile uint32_t *dma = (uint32_t *)mmap(NULL, DMA_REG_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, dev_mem, AXI_DMA_BASE);
    volatile int16_t *tx_buf = (volatile int16_t *)mmap(NULL, BUF_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, dev_mem, TX_BUF_PADDR);
    volatile int16_t *rx_buf = (volatile int16_t *)mmap(NULL, BUF_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, dev_mem, RX_BUF_PADDR);

    if (load_fc_weights(".") != 0) {
        printf("[FATAL] Missing binary weight files.\n"); return EXIT_FAILURE;
    }

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    int opt = 1; setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    inet_pton(AF_INET, BIND_IP, &servaddr.sin_addr);
    bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr));

    dma_reset(dma);
    uint8_t raw_pixel_frame[IMG_PIXELS];
    
    printf("[READY] ZedBoard CNN Pipeline armed.\n\n");

    while (1) {
        ssize_t len = recvfrom(sockfd, raw_pixel_frame, sizeof(raw_pixel_frame), 0, NULL, NULL);
        if (len != IMG_PIXELS) continue;

        for (int i = 0; i < IMG_PIXELS; i++) {
            tx_buf[i] = (int16_t)(((uint16_t)raw_pixel_frame[i] * 256) / 255);
        }

        __sync_synchronize();

        dma_set(dma, S2MM_DMACR, 0x01);
        dma_set(dma, S2MM_DA, RX_BUF_PADDR);
        dma_set(dma, S2MM_LENGTH, OUT_DMA_BYTES);

        dma_set(dma, MM2S_DMACR, 0x01);
        dma_set(dma, MM2S_SA, TX_BUF_PADDR);
        dma_set(dma, MM2S_LENGTH, IN_DMA_BYTES);

        int timeout_tx = 10000000;
        while (!(dma_get(dma, MM2S_DMASR) & 0x02) && --timeout_tx);
        if (timeout_tx == 0) {
            printf("\n[TIMEOUT] MM2S stalled! Resetting DMA...\n");
            dma_reset(dma);
            continue;
        }

        int timeout_rx = 10000000;
        while (!(dma_get(dma, S2MM_DMASR) & 0x02) && --timeout_rx);
        if (timeout_rx == 0) {
            printf("\n[WARN] S2MM missing TKEEP/TLAST. Force-reading DDR...\n");
            dma_reset(dma); 
            // Intentionally no 'continue' here so it extracts features anyway
        }

        __sync_synchronize();
        int prediction = run_arm_classifier((const int16_t *)rx_buf);
        printf("========================================\n");
        printf("[RESULT] Detected Digit: >>> %d <<<\n", prediction);
        printf("========================================\n\n");
    }

    close(sockfd);
    close(dev_mem);
    return EXIT_SUCCESS;
}