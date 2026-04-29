# Matrix Multiplication optimized

## Overview
This project uses NVIDIA cuBLAS to perform optimized matrix multiplications. It is designed to demonstrate high-performance computing (HPC) capabilities.

## Repository Contents
- **/src/main.cu**: Core CUDA source code.
- **Makefile**: Professional build script for nvcc.
- **run.sh**: Automation script for multi-size testing.
- **/data/output_log.txt**: Proof of successful execution on various datasets.

## How to Build and Run
1. **Compile**: Type `make` in the terminal.
2. **Automated Test**: Type `./run.sh` to see the code run on Small, Medium, and Large matrices.
3. **Manual Run**: `./bin/matrix_transformer <size>` (e.g., `./bin/matrix_transformer 1024`).
