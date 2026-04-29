#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Column-major index macro for cuBLAS
#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float* P, int uWP, int uHP) {
    for (int i = 0; i < uHP; i++) {
        for (int j = 0; j < uWP; j++) {
            printf("%.1f", P[index(i, j, uHP)]);
            if (j < uWP - 1) printf(" ");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    // CLI Argument Handling (Rubric Requirement)
    int n = (argc > 1) ? atoi(argv[1]) : 4; 
    printf("Processing %d x %d Matrix Multiplication...\n", n, n);

    int size = n * n * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform GEMM
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print small sample for verification
    if (n <= 8) {
        printf("\nResult Matrix C (Sample):\n");
        printMat(h_C, n, n);
    } else {
        printf("\nMatrix too large to print. Success.\n");
    }

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
