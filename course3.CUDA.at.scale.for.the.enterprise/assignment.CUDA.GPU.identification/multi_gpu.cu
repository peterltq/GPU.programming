#include <stdio.h>

// Based on https://cuda-programming.blogspot.com/2013/01/how-to-query-device-properties-and.html
int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  printf("Number of GPU Devices: %d\n", nDevices);

  // You will need to track the minimum or maximum for one or more device properties, so initialize them here
  int currentChosenDeviceNumber = -1; // Will not choose a device by default 
  
  cudaDeviceProp props[32];
    for (int i = 0; i < nDevices && i < 32; i++) {
    //cudaDeviceProp prop;
    cudaGetDeviceProperties(&props[i], i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", props[i].name);
    printf("  Device Compute Major: %d Minor: %d\n", props[i].major, props[i].minor);
    printf("  Max Thread Dimensions: [%d][%d][%d]\n", props[i].maxThreadsDim[0], props[i].maxThreadsDim[1], props[i].maxThreadsDim[2]);
    printf("  Max Threads Per Block: %d\n", props[i].maxThreadsPerBlock);
    printf("  Number of Multiprocessors: %d\n", props[i].multiProcessorCount);
    printf("  Device Clock Rate (KHz): %d\n", props[i].clockRate);
    printf("  Memory Bus Width (bits): %d\n", props[i].memoryBusWidth);
    printf("  Registers Per Block: %d\n", props[i].regsPerBlock);
    printf("  Registers Per Multiprocessor: %d\n", props[i].regsPerMultiprocessor);
    printf("  Shared Memory Per Block: %zu\n", props[i].sharedMemPerBlock);
    printf("  Shared Memory Per Multiprocessor: %zu\n", props[i].sharedMemPerMultiprocessor);
    printf("  Total Constant Memory (bytes): %zu\n", props[i].totalConstMem);
    printf("  Total Global Memory (bytes): %zu\n", props[i].totalGlobalMem);
    printf("  Warp Size: %d\n", props[i].warpSize);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*props[i].memoryClockRate*(props[i].memoryBusWidth/8)/1.0e6);
    // You can set the current chosen device property based on tracked min/max values
    
  }

  for (int j = 0; j < nDevices && j < 32; j++) {
    if (props[j].multiProcessorCount > 32) {
      currentChosenDeviceNumber = j;
      break;
    }
  }

  // Create logic to actually choose the device based on one or more device properties

  // Print out the chosen device as below
  printf("The chosen GPU device has an index of: %d\n",currentChosenDeviceNumber); 

  return 0;
}