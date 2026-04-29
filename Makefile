# Use nvcc to compile CUDA code
CC = nvcc
# Link the cuBLAS library
CFLAGS = -lcublas
# Output location
TARGET = bin/matrix_transformer

all:
	mkdir -p bin
	$(CC) src/main.cu -o $(TARGET) $(CFLAGS)

clean:
	rm -rf bin
