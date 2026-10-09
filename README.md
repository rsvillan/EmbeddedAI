# EmbeddedAI
Using an FPGA and Raspberry pi to build a custom AI image recognizer in hardware.

# Edge AI Video Inference Accelerator

**Course:** CSI-4130/5130: Artificial Intelligence (Oakland University)
**Author:** Roger Villanueva

## Problem Statement

Deploying robust artificial intelligence models, such as Convolutional Neural Networks (CNNs) for real-time video processing, is computationally expensive and often bottlenecked by standard CPU architectures on edge devices. This project tackles the challenge of edge AI acceleration by offloading neural network inference to custom hardware, specifically utilizing a Field Programmable Gate Array (FPGA) to achieve lower latency and higher throughput than purely software-based solutions.

## Proposed Method

This project develops an end-to-end hardware-accelerated inference pipeline bridging a standard edge device and an FPGA:

1. **Data Capture & Streaming:** A Raspberry Pi 4B captures live video frames via a connected MIPI camera and transmits the data over an Ethernet connection using standard TCP/UDP sockets.
2. **Linux-Based Orchestration:** A ZedBoard FPGA development board running a custom PetaLinux image receives the network stream. A user-space application writes the incoming frames into contiguous memory.
3. **Hardware Acceleration (Custom IP):** The PetaLinux environment triggers an AXI Direct Memory Access (DMA) transfer, pushing the video data through a custom PL (Programmable Logic) IP block. Initially utilizing a simple AXI-Stream adder for pipeline validation, this custom IP is being expanded into a dedicated AI inference accelerator (e.g., matrix multiplication units for CNN layers) designed using Verilog and AMD Vivado.
4. **Model Prototyping:** The underlying AI model weights and architecture will be developed and trained using PyTorch within a Windows 11 environment (using Python 3.14 managed via UV) before synthesizing the inference logic for the FPGA.

## Data Sources

* **Live Inference Data:** Real-time video stream captured directly from the Raspberry Pi 4B MIPI camera module.
* **Training Data:** [Insert Dataset Name, e.g., CIFAR-10, COCO, or a custom image dataset] used offline in PyTorch to train the model prior to hardware deployment.

## Setup and Execution Instructions

*(Detailed instructions will be added as the project progresses toward the final submission on November 30, 2026).*

### Prerequisites

* **Hardware:** Raspberry Pi 4B, ZedBoard FPGA, MIPI Camera.
* **Software:** PetaLinux, AMD Vivado, VS Code, Python 3.14 (uv environment), PyTorch.

### Running the Project

1. [Placeholder: Instructions to flash and boot PetaLinux on the ZedBoard]
2. [Placeholder: Instructions to start the Python streaming script on the Raspberry Pi]
3. [Placeholder: Instructions to initialize the AXI DMA transfer via `/dev/uio` or `udmabuf`]

## Video Presentation

[Placeholder: Link to the final recorded demo and presentation]

## Citations & Acknowledgements

* Oakland University CSI-4130/5130 Course Materials.
* [Placeholder: Any open-source Verilog modules, PyTorch repositories, or research papers referenced for the CNN architecture.]
* [Placeholder: Any external libraries used, e.g., Scikit-learn, SciPy, NumPy.]