#ifndef CPU_GPU_MEMORY_H
#define CPU_GPU_MEMORY_H

#include <stdio.h>
#include <tuple>
#include <bits/stdc++.h>
#include <string>
#include <fstream>
#include <vector>
#include <utility>   // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream>   // std::stringstream
#include <ctime>

using namespace std;

// For the CUDA runtime routines (prefixed with "cuda_")
__constant__ int d_v;

// Kernel declaration
__global__ void add(int *d_a, int *d_b, int *d_c, int numElements);

// Host function declarations (use std::tuple return types explicitly)
__host__ std::tuple<int *, int *, int *> allocateRandomHostMemory(int numElements);
__host__ std::tuple<int *, int *, int *> allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDevice(int *h_a, int *h_b, int *d_a, int *d_b, int numElements);
__host__ void executeKernel(int *d_a, int *d_b, int *d_c, int numElements, int threadsPerBlock);
__host__ void deallocateMemory(int *d_a, int *d_b, int *d_c);
__host__ void cleanUpDevice();
__host__ std::tuple<int, int> parseCommandLineArguments(int argc, char *argv[]);

#endif // CPU_GPU_MEMORY_H
