// Self-contained radix_sort.cu
// Original by Bulat Ziganshin, modified to remove CUDA Samples dependencies
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <string.h>

// Parameters
const int defaultNumElements = 16 << 20;
double MIN_BENCH_TIME = 0.5;  // minimum seconds to run each benchmark

// ----------------------
// Helper macros
// ----------------------
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<unsigned int>(result),
                cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// ----------------------
// Device utility functions
// ----------------------
template <typename T>
__global__ void fill_with_random(T *d_array, size_t size) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    uint32_t rnd = idx * 1234567891u;
    rnd = 29943829 * rnd + 1013904223;
    rnd = 29943829 * rnd + 1013904223;
    uint64_t rnd1 = rnd;
    rnd = 29943829 * rnd + 1013904223;
    rnd = 29943829 * rnd + 1013904223;
    d_array[idx] = (T)((rnd1 << 32) + rnd);
}

// ----------------------
// Sorting functions
// ----------------------
template <typename Key>
double key_sort(int SORT_BYTES, size_t n, void *d_array0, cudaEvent_t &start, cudaEvent_t &stop) {
    int begin_bit = 0, end_bit = SORT_BYTES * 8;
    Key *d_array = (Key*)d_array0;

    cub::DoubleBuffer<Key> d_keys(d_array, d_array + n);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    checkCudaErrors(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, n, begin_bit, end_bit));
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    int numIterations = 0;
    double totalTime = 0;

    while (totalTime < MIN_BENCH_TIME) {
        fill_with_random<Key><<<n/1024 + 1, 1024>>>(d_array, n);
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaEventRecord(start, nullptr));
        checkCudaErrors(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, n, begin_bit, end_bit));
        checkCudaErrors(cudaEventRecord(stop, nullptr));
        checkCudaErrors(cudaDeviceSynchronize());

        float elapsed;
        checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
        totalTime += elapsed / 1000.0;
        numIterations++;
    }

    checkCudaErrors(cudaFree(d_temp_storage));
    return totalTime / numIterations;
}

template <typename Key, typename Value>
double keyval_sort(int SORT_BYTES, size_t n, void *d_array0, cudaEvent_t &start, cudaEvent_t &stop) {
    int begin_bit = 0, end_bit = SORT_BYTES * 8;
    Key *d_array = (Key*)d_array0;
    Value *d_value_array = (Value*)(d_array + 2 * n);

    cub::DoubleBuffer<Key> d_keys(d_array, d_array + n);
    cub::DoubleBuffer<Value> d_values(d_value_array, d_value_array + n);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n, begin_bit, end_bit));
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    int numIterations = 0;
    double totalTime = 0;

    while (totalTime < MIN_BENCH_TIME) {
        fill_with_random<Key><<<n/1024 + 1, 1024>>>(d_array, n);
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaEventRecord(start, nullptr));
        checkCudaErrors(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n, begin_bit, end_bit));
        checkCudaErrors(cudaEventRecord(stop, nullptr));
        checkCudaErrors(cudaDeviceSynchronize());

        float elapsed;
        checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
        totalTime += elapsed / 1000.0;
        numIterations++;
    }

    checkCudaErrors(cudaFree(d_temp_storage));
    return totalTime / numIterations;
}

// ----------------------
// Main
// ----------------------
int main(int argc, char **argv) {
    bool full = false;
    int numElements = defaultNumElements;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "full") == 0) full = true;
        else numElements = atoi(argv[i]);
    }

    if (numElements < 16384) numElements <<= 20;

    // Display CUDA devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);

    // Allocate maximum memory for largest type (uint64_t)
    size_t maxElementSize = sizeof(uint64_t);
    void* d_array;
    checkCudaErrors(cudaMalloc(&d_array, 4 * numElements * maxElementSize));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    auto print = [&](int bytes, int keysize, int valsize, double totalTime) {
        char valsize_str[100];
        sprintf(valsize_str, (valsize ? "+%d" : "  "), valsize);
        printf("%d/%d%s: Throughput =%9.3lf MElements/s, Time = %.3lf ms\n",
               bytes, keysize, valsize_str, 1e-6 * numElements / totalTime, totalTime * 1000);
    };

    printf("Sorting %dM elements:\n", numElements >> 20);

    // Key-only sorts
    if (full) { for(int i=1;i<=1;i++) print(1,1,0,key_sort<uint8_t>(1,numElements,(void*)d_array,start,stop)); printf("\n"); }
    if (full) { for(int i=1;i<=2;i++) print(2,2,0,key_sort<uint16_t>(2,numElements,(void*)d_array,start,stop)); printf("\n"); }
              { for(int i=1;i<=4;i++) print(4,4,0,key_sort<uint32_t>(4,numElements,(void*)d_array,start,stop)); printf("\n"); }
              { for(int i=1;i<=8;i++) print(8,8,0,key_sort<uint64_t>(8,numElements,(void*)d_array,start,stop)); printf("\n"); }

    // Key-value sorts
    if (full) { for(int i=1;i<=1;i++) print(1,1,1,keyval_sort<uint8_t,uint8_t>(1,numElements,(void*)d_array,start,stop)); printf("\n"); }
    if (full) { for(int i=1;i<=1;i++) print(1,1,2,keyval_sort<uint8_t,uint16_t>(1,numElements,(void*)d_array,start,stop)); printf("\n"); }
    if (full) { for(int i=1;i<=1;i++) print(1,1,4,keyval_sort<uint8_t,uint32_t>(1,numElements,(void*)d_array,start,stop)); printf("\n"); }
    if (full) { for(int i=1;i<=1;i++) print(1,1,8,keyval_sort<uint8_t,uint64_t>(1,numElements,(void*)d_array,start,stop)); printf("\n"); }

    return 0;
}
