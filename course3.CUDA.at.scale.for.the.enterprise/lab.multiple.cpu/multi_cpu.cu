#include <stdio.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <stdlib.h>

// Function to parse CSV line into float array
void parseFloatsToArrayFromString(float *arr, std::string line, int numElements)
{
    size_t pos = 0;
    for (int i = 0; i < numElements; i++)
    {
        size_t comma = line.find(',', pos);
        std::string token = line.substr(pos, comma - pos);
        arr[i] = std::stof(token);
        pos = comma + 1;
    }
}

// Host function to retrieve input data from CSV files
__host__ void retrieveDataFromFiles(float *h_a, float *h_b, int numElements)
{
    printf("Retrieving data from input files.\n");

    // Clean up old lock files to prevent re-entry
    remove("./output_a.lock");
    remove("./output_b.lock");

    // Wait for input CSVs to exist
    while (access("./input_a.csv", F_OK) == -1 || access("./input_b.csv", F_OK) == -1)
    {
        printf("Waiting for input CSV files to appear...\n");
        sleep(2);
    }

    printf("Parsing array from input csv files.\n");

    std::string line_a, line_b;
    std::ifstream file_a("./input_a.csv");
    std::ifstream file_b("./input_b.csv");

    if (file_a.is_open() && file_b.is_open())
    {
        getline(file_a, line_a);
        printf("Parsing line from input_a.csv\n");
        parseFloatsToArrayFromString(h_a, line_a, numElements);

        getline(file_b, line_b);
        printf("Parsing line from input_b.csv\n");
        parseFloatsToArrayFromString(h_b, line_b, numElements);
    }

    // Remove input locks after reading data
    remove("./input_a.lock");
    remove("./input_b.lock");
}

// Dummy kernel for example
__global__ void vectorDiffKernel(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        c[idx] = a[idx] - b[idx];
}

// Host function to perform a single CPU/CUDA iteration
__host__ void performMultiCPUIteration()
{
    // Cleanup stale lock files before running
    remove("./input_a.lock");
    remove("./input_b.lock");
    remove("./output_a.lock");
    remove("./output_b.lock");

    int numElements = 128;
    printf("Vector difference of %d elements\n", numElements);

    float *h_a = (float *)malloc(numElements * sizeof(float));
    float *h_b = (float *)malloc(numElements * sizeof(float));
    float *h_c = (float *)malloc(numElements * sizeof(float));

    retrieveDataFromFiles(h_a, h_b, numElements);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, numElements * sizeof(float));
    cudaMalloc(&d_b, numElements * sizeof(float));
    cudaMalloc(&d_c, numElements * sizeof(float));

    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_a, h_a, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    printf("CUDA kernel launch with 1 blocks of 256 threads\n");
    vectorDiffKernel<<<1, 256>>>(d_a, d_b, d_c, numElements);
    cudaDeviceSynchronize();
    printf("Completed execution kernel\n");

    // Copy result back
    cudaMemcpy(h_c, d_c, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Save output files
    printf("Placing calculation results into output files\n");
    std::ofstream out("./output_a.csv");
    for (int i = 0; i < numElements; i++)
        out << h_c[i] << (i == numElements - 1 ? "\n" : ",");
    out.close();

    // Cleanup device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Cleanup host memory
    free(h_a);
    free(h_b);
    free(h_c);
}

int main()
{
    performMultiCPUIteration();
    cudaDeviceReset();
    printf("Done\n");
    return 0;
}
