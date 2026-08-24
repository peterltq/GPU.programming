# Thrust Lab Activity

This lab introduces learners to the basics of the Thrust library in CUDA. You'll explore how to use `thrust::host_vector` and `thrust::device_vector` to manage memory and perform simple operations on the GPU without writing custom kernels.

---

## Objectives

- Understand the difference between host and device vectors
- Resize and initialize vectors dynamically
- Copy data between host and device using Thrust
- Modify device memory directly
- Print and verify vector contents

---

## Key Concepts

- Thrust Library
- Host vs Device Memory
- Vector Initialization and Resizing
- GPU Memory Abstraction
- CUDA C++ with Modern Standards

---

## How to Compile

Open the terminal in your lab environment using the shortcut "Ctrl + Shift + `(backtick)", then run the following command:

nvcc --std=c++14 thrust_example.cu -o thrust_example

---

## How to Run

After compiling, run the executable:

./thrust_example

---

## Expected Output

H has size 4
H[0] = 14
H[1] = 20
H[2] = 38
H[3] = 46
H now has size 2
D[0] = 99
D[1] = 88