# 🚀 CUDA Streams and Concurrency Lab

## 🎯 Lab Overview
This lab demonstrates the use of **CUDA Streams** to achieve **asynchronous and concurrent execution** of memory transfers and kernel operations on the GPU.  
Using multiple CUDA streams allows overlapping data transfers and computations, improving GPU utilization and reducing overall runtime.

The program (`streams.cu`) performs simple arithmetic operations on arrays using multiple GPU kernels and compares different stream execution strategies.

---

## 🧩 Learning Objectives
By completing this lab, you will be able to:
- Understand the concept of **CUDA streams** and how they enable **concurrent execution**.
- Implement **asynchronous memory transfers** between host and device using streams.
- Launch multiple **independent GPU kernels** in parallel.
- Compare performance differences between **full asynchronous execution** and **blocking stream** strategies.
- Manage device memory and synchronization effectively in CUDA programs.

---

## ⚙️ Lab Components

| Component | Description |
|------------|-------------|
| `kernelA1` | Increments each array element by 1 |
| `kernelB1` | Doubles each array element |
| `kernelA2` | Decrements each array element by 1 |
| `kernelB2` | Halves each array element |
| `allocateHostMemory()` | Allocates pinned host memory and initializes it with random floats |
| `allocateDeviceMemory()` | Allocates device memory on the GPU |
| `copyFromHostToDeviceSync()` / `Async()` | Copies data from host to device (synchronously/asynchronously) |
| `copyFromDeviceToHostSync()` / `Async()` | Copies data from device to host (synchronously/asynchronously) |
| `runStreamsFullAsync()` | Runs all kernels in parallel across independent streams |
| `runStreamsBlockingKernel2StreamsNaive()` | Runs kernels in two streams with partial blocking |
| `runStreamsBlockingKernel2StreamsOptimal()` | Optimized ordering of kernels across two streams |
| `printHostMemory()` | Prints the resulting host memory values for verification |

---

## Theory Recap

A **CUDA stream** is a sequence of operations that execute in order on the device, but different streams can execute concurrently.  
When using pinned (page-locked) host memory and asynchronous operations (`cudaMemcpyAsync` and kernel launches with stream arguments), CUDA can overlap **data transfer** and **kernel execution**, enabling true **concurrency**.

---
## Compile the code
nvcc -std=c++17 -o streams streams.cu

## Run the code
./streams

## 🧱 Code Structure

```plaintext
streams.cu
├── Kernel definitions (A1, B1, A2, B2)
├── Memory management functions
├── Data transfer (sync & async)
├── Stream management and kernel launch functions
├── Three main execution modes:
│   ├── runStreamsFullAsync()
│   ├── runStreamsBlockingKernel2StreamsNaive()
│   └── runStreamsBlockingKernel2StreamsOptimal()
└── main() – Runs all three methods and prints results
```plaintext

