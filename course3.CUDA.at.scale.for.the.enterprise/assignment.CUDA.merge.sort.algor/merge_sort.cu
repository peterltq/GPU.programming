#include "merge_sort.h"
#include <random>
#include <cuda_runtime.h>
#include <tuple>
#include <iostream>
#include <string>

#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)
// Based on https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu


// 
// Get the time (in microseconds) since the last call to tm();
// the first value returned by this must not be trusted
//
timeval tStart;
int tm() {
    timeval tEnd;
    gettimeofday(&tEnd, 0);
    int t = (tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec;
    tStart = tEnd;
    return t;
}

__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc, char** argv) 
{
    int numElements = 32;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 128;
    ///threadsPerBlock.x = 1;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    //blocksPerGrid.x = 1;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'n':
                    i++;
                    numElements = std::stoi(argv[i]);
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    return {threadsPerBlock, blocksPerGrid, numElements};
}

__host__ long *generateRandomLongArray(int numElements)
{
    //TODO generate random array of long integers of size numElements
    long *randomLongs = (long *) malloc(numElements * sizeof(long));

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<long> dist(0, 1000);

    for (int i = 0; i < numElements; ++i) {
        randomLongs[i] = dist(gen);
    }

    return randomLongs;
}

__host__ void printHostMemory(long *host_mem, int num_elments)
{
    // Output results
    for(int i = 0; i < num_elments; i++)
    {
        printf("%ld ",host_mem[i]);
    }
    printf("\n");
}

__device__ void printDeviceMemory(long *dev_mem, long start, long len)
{
    printf("printDeviceMemory: start:%ld, len:%ld\n", start, len);
    // Output results
    
    for(long i = 0; i < len; i++)
    {
        printf("dev_mem[%ld] = %ld, ", start + i, dev_mem[start + i]);
    }
    printf("\n");
}


__host__ std::tuple <long* ,long* ,dim3* ,dim3*> allocateMemory(int numElements, dim3 threadsPerBlock, dim3 blocksPerGrid)
{
    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    checkCudaErrors(cudaMalloc((void**) &D_data, numElements * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**) &D_swp, numElements * sizeof(long)));

    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

    // Actually allocate the two arrays


    // Copy from our input list into the first array
 
    // Copy the thread / block info to the GPU as well
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    return {D_data, D_swp, D_threads, D_blocks};
}

__host__ long* mergesort(long* data, long numElements, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    auto[D_data, D_swp, D_threads, D_blocks] = allocateMemory(numElements, threadsPerBlock, blocksPerGrid);
    checkCudaErrors(cudaMemcpy(D_data, data, numElements * sizeof(long), cudaMemcpyHostToDevice));

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;
    printf("nThreads: %ld\n", nThreads);

    // TODO Initialize timing metrics variable(s). The implementation of this is up to you
    tm();
    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    bool resultInSwp = false;
    for (long width = 2; width < (numElements << 1); width <<= 1) {
        //long numSlices = max((numElements / (nThreads * width)), 1);
        long numSlices = numElements / ((nThreads) * width) + 1;
        printf("width:%ld, numSlices:%ld\n", width, numSlices);

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, numElements, width, numSlices, D_threads, D_blocks); 

        // Switch the input / output arrays instead of copying them around
        long * tmp = A;
        A = B;
        B = tmp;
        resultInSwp = !resultInSwp;
    }

    long * result = (resultInSwp) ? D_swp : D_data;
    checkCudaErrors(cudaMemcpy(data, result, numElements * sizeof(long), cudaMemcpyDeviceToHost));
    
    // TODO calculate and print to stdout kernel execution time
    std::cout << "Time taken: " << tm() << " microseconds.\n";

    // Free the GPU memory
    cudaFree(D_data);
    cudaFree(D_swp);
    cudaFree(D_threads);
    cudaFree(D_blocks);
    
    return data;
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    // TODO initialize 3 long variables start, middle, and end
    // middle and end do not have values set,
    // while start is set to the width of the merge sort data span * the thread index * number of slices that this kernel will sort
    long start = idx * width * slices;
    long middle; 
    long end;

    ///printf("gpu_mergesort: idx:%d, start:%ld, width:%ld, slices:%ld\n", idx, start, width, slices);

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;
        
        middle = min(start + (width >> 1), size);
        end = min(start + width, size);

        gpu_bottomUpMerge(source, dest, start, middle, end);

        start += width;    

        // Break from loop when the start variable is >= size of the input array

        // Set middle to be minimum middle index (start index plus 1/2 width) and the size of the input array

        // Set end to the minimum of the end index (start index plus the width of the current data window) and the size of the input array
       
        // Perform bottom up merege given the two available arrays and the start, middle, and end variables
        
        // Increase the start index by the width of the current data window
    }

}

//
// Finally, sort something gets called by gpu_mergesort() for each slice
// Note that the pseudocode below is not necessarily 100% complete you may want to review the merge sort algorithm.
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    ///printf("gpu_bottomUpMerge: start:%ld, mid:%ld, end:%ld\n", start, middle, end);
    long i = start;
    long j = middle;

    // Create a for look that iterates between the start and end indexes
    for (int k = start; k < end; ++k) {
        // if i is before the middle index and (j is the final index or the value at i <  the value at j)
        if (i < middle && (j >= end || source[i] < source[j])) {
            // set the value in the destination array at index k to the value at index i in the source array
            dest[k] = source[i];
            i++;
            // increment i
        } else {
            // set the value in the destination array at index k to the value at index j in the source array
            dest[k] = source[j];
            j++;    
            // increment k
        }
    }
    ///printf("gpu_bottomUpMerge: After one iteration:\n");
    ///printDeviceMemory(dest, start, end - start);

}

__host__ int main(int argc, char** argv) 
{
    tm();
    auto[threadsPerBlock, blocksPerGrid, numElements] = parseCommandLineArguments(argc, argv);

    long *data = generateRandomLongArray(numElements);

    printf("Unsorted data: ");
    printHostMemory(data, numElements);

    data = mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

    printf("Sorted data: ");
    printHostMemory(data, numElements);
}