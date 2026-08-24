
---

## Generating Input Data
You can generate sample input vectors using Python:

## Steps to perform the lab

# Remove any previous lock files
rm -f input_a.lock input_b.lock output_a.lock output_b.lock

# Generate random 128-element input vectors
python3 -c "import random; open('input_a.csv','w').write(','.join(f'{random.random():.6f}' for _ in range(128)))"
python3 -c "import random; open('input_b.csv','w').write(','.join(f'{random.random():.6f}' for _ in range(128)))"

# Create lock files to signal inputs are ready
touch input_a.lock input_b.lock

# Compile the code
nvcc -o multi_cpu multi_cpu.cpp

# Execute the lab
./multi_cpu
