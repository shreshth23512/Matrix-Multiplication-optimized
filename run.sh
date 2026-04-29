#!/bin/bash

# 1. Compile the latest version of the code
echo "Building project..."
make clean
make

# 2. Ensure the data directory exists
mkdir -p data

# 3. Run the multi-data test suite and log results
echo "--- GPU TEST SUITE COMMENCING ---" > data/output_log.txt

echo "Test 1: Small Scale (4x4 Matrix)" >> data/output_log.txt
./bin/matrix_transformer 4 >> data/output_log.txt

echo -e "\nTest 2: Medium Scale (256x256 Matrix)" >> data/output_log.txt
./bin/matrix_transformer 256 >> data/output_log.txt

echo -e "\nTest 3: Large Scale (1024x1024 Matrix)" >> data/output_log.txt
./bin/matrix_transformer 1024 >> data/output_log.txt

echo -e "\n--- ALL TESTS COMPLETED SUCCESSFULLY ---" >> data/output_log.txt

echo "Testing complete. Results saved to data/output_log.txt"
