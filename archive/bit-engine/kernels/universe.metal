#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// KERNEL 1: FIND ROW MAX (Reduction)
// -----------------------------------------------------------------------------
// Calculates the maximum absolute value for each row (vector) in the tensor.
// This determines the 'scale' used for symmetric quantization.
//
// Grid Strategy: 
// - Dispatch 1 Threadgroup per Row.
// - Threads within the group perform parallel reduction.
// -----------------------------------------------------------------------------
kernel void find_row_max(
    device const float* src     [[ buffer(0) ]], // Input Tensor [Rows * Dim]
    device float* scales        [[ buffer(1) ]], // Output Scales [Rows]
    constant uint& dim          [[ buffer(2) ]], // Dimension size (e.g., 4096)
    uint2 tgid                  [[ threadgroup_position_in_grid ]],
    uint2 tid                   [[ thread_position_in_threadgroup ]],
    uint2 threadsPerGroup       [[ threads_per_threadgroup ]]
) {
    uint row = tgid.x;
    uint thread_idx = tid.x;
    uint block_size = threadsPerGroup.x;

    // Pointer to the start of this row
    device const float* row_ptr = src + (row * dim);

    // 1. Thread-Local Max
    // Each thread strides through the row to handle dimensions > block_size
    float local_max = 0.0f;
    for (uint i = thread_idx; i < dim; i += block_size) {
        float val = abs(row_ptr[i]);
        if (val > local_max) local_max = val;
    }

    // 2. Threadgroup Reduction (Shared Memory)
    // We use a simplified SIMD-group reduction if available, or just shared mem.
    // Here we use threadgroup memory for compatibility.
    threadgroup float shared_max[1024]; // Max threads per group assumption
    shared_max[thread_idx] = local_max;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Standard Tree Reduction
    for (uint s = block_size / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            shared_max[thread_idx] = max(shared_max[thread_idx], shared_max[thread_idx + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Write Result (Thread 0 only)
    if (thread_idx == 0) {
        // Prevent division by zero later: clamp min scale
        float m = shared_max[0];
        if (m < 1e-9f) m = 1.0f; 
        
        // Calculate scale: max_val / 127.0
        scales[row] = m / 127.0f;
    }
}

// -----------------------------------------------------------------------------
// KERNEL 2: QUANTIZE (Apply)
// -----------------------------------------------------------------------------
// Applies the scale to convert F32 -> Int8.
//
// Grid Strategy:
// - Flat dispatch over all elements (Rows * Dim).
// -----------------------------------------------------------------------------
kernel void quantize_f32_int8(
    device const float* src     [[ buffer(0) ]], // Input Tensor
    device const float* scales  [[ buffer(1) ]], // Pre-calculated Scales per row
    device char* dst            [[ buffer(2) ]], // Output Int8 Buffer
    constant uint& dim          [[ buffer(3) ]], // Dimension size
    uint id                     [[ thread_position_in_grid ]]
) {
    // 1. Identify Row
    uint row = id / dim;
    
    // 2. Fetch Scale
    float scale = scales[row];
    
    // 3. Fetch Value
    float val = src[id];
    
    // 4. Quantize
    // Formula: round(val / scale) -> clamp(-127, 127)
    float q = round(val / scale);
    
    // Clamp to int8 range
    if (q > 127.0f) q = 127.0f;
    if (q < -127.0f) q = -127.0f;
    
    // 5. Write
    dst[id] = (char)q;
}