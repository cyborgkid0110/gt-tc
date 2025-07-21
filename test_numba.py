import numpy as np
from numba import cuda
import time

# cuda.close()  # Close context CUDA

# Define a CUDA kernel
@cuda.jit
def add_arrays(a, b, c):
    # Get the thread's index
    idx = cuda.grid(1)  # 1D grid of threads
    if idx < c.size:    # Ensure we don't go out of bounds
        c[idx] = a[idx] + b[idx]

# Create input arrays
n = 10000000
a = np.random.random(n).astype(np.float32)  # Random floats
b = np.random.random(n).astype(np.float32)  # Random floats
c = np.zeros(n, dtype=np.float32)           # Output array

# Copy arrays to the GPU
a_device = cuda.to_device(a)
b_device = cuda.to_device(b)
c_device = cuda.to_device(c)

# Configure the grid and block dimensions
threads_per_block = 512
# Calculate minimum blocks to handle all threads.
blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

# Launch the kernel
start = time.time()
add_arrays[blocks_per_grid, threads_per_block](a_device, b_device, c_device)
cuda.synchronize()  # Wait for the kernel to finish
print(f"Kernel execution time: {time.time() - start:.4f} seconds")

# Copy the result back to the host
c_result = c_device.copy_to_host()

# Verify the result
np.testing.assert_almost_equal(c_result, a + b)
print("First 5 elements of result:", c_result[:5])

a_device = None
b_device = None
c_device = None
cuda.current_context().deallocations.clear()

