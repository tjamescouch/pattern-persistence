#include "gpu/metal/pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

using namespace std;

// ====================================================================
// C-API EXPORTS (For Python FTT Integration)
// ====================================================================
extern "C" {

// 1. Initialization Hook: Initializes Metal device and loads kernels.
// Returns an opaque pointer to the allocated BitPipeline object.
void* init_bit_pipeline() {
    try {
        BitPipeline* pipe = new BitPipeline();
        pipe->init();
        return (void*)pipe;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Pipeline initialization failed: " << e.what() << endl;
        return nullptr;
    }
}

// 2. Quantization Hook: Executes the two-pass quantization kernel.
// Takes host memory pointers directly.
void quantize_batch(void* pipe_ptr, const float* src, char* dst, float* scales, uint32_t rows, uint32_t dim) {
    if (!pipe_ptr) {
        std::cerr << "[ERROR] Pipeline pointer is null." << endl;
        return;
    }
    BitPipeline* pipe = (BitPipeline*)pipe_ptr;
    pipe->dispatch_quantize(src, dst, scales, rows, dim);
}

// 3. Cleanup Hook: Deallocates the pipeline object.
void cleanup_bit_pipeline(void* pipe_ptr) {
    if (pipe_ptr) {
        delete (BitPipeline*)pipe_ptr;
    }
}

} // extern "C"

// ====================================================================
// STANDALONE TEST HARNESS
// ====================================================================

int main(int argc, char** argv) {
    // Test parameters: 10 rows (vectors) of 4096 dimensions
    const uint32_t TEST_ROWS = 10;
    const uint32_t TEST_DIM = 4096;
    const size_t SIZE = TEST_ROWS * TEST_DIM;

    // --- 1. SETUP HOST MEMORY ---
    std::vector<float> src_data(SIZE);
    std::vector<char> dst_data(SIZE);
    std::vector<float> scales_data(TEST_ROWS);

    // Initialize with random data (Gaussian distribution, symmetric around 0)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 1.0f); // Mean 0, Std Dev 1.0

    for (size_t i = 0; i < SIZE; ++i) {
        src_data[i] = d(gen) * 5.0f; // Scale up the variance for better test data
    }

    // --- 2. EXECUTE METAL PIPELINE (via C-API) ---
    void* pipe_ptr = init_bit_pipeline();
    if (!pipe_ptr) return 1;

    try {
        std::cout << "[Test] Dispatching " << TEST_ROWS << "x" << TEST_DIM << " quantization job..." << endl;
        
        quantize_batch(pipe_ptr, 
                       src_data.data(), 
                       dst_data.data(), 
                       scales_data.data(), 
                       TEST_ROWS, 
                       TEST_DIM);

        std::cout << "[Test] Job complete. Validating results." << endl;

        // --- 3. VALIDATE OUTPUT ---
        // Check 1: Scales must be positive and within reasonable bounds
        size_t non_zero_scales = 0;
        float max_scale = 0.0f;
        for (uint32_t r = 0; r < TEST_ROWS; ++r) {
            if (scales_data[r] > 1e-9f) non_zero_scales++;
            if (scales_data[r] > max_scale) max_scale = scales_data[r];
        }

        // Check 2: Int8 data should reflect quantization (check one extreme point)
        char q_max = 0;
        for (size_t i = 0; i < SIZE; ++i) {
            if (std::abs(dst_data[i]) > std::abs(q_max)) q_max = dst_data[i];
        }

        std::cout << "--- Validation ---" << endl;
        std::cout << "Scales found: " << non_zero_scales << "/" << TEST_ROWS << endl;
        std::cout << "Max Scale:    " << max_scale << endl;
        std::cout << "Max Quantized:| " << (int)q_max << " | (Target: ~127)" << endl;
        
        if (non_zero_scales == TEST_ROWS && std::abs(q_max) >= 100) {
            std::cout << "✅ TEST HARNESS PASS: Output appears correctly quantized." << endl;
        } else {
            std::cout << "❌ TEST HARNESS FAIL: Quantization did not saturate correctly." << endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Execution exception: " << e.what() << endl;
        return 1;
    }

    // --- 4. CLEANUP ---
    cleanup_bit_pipeline(pipe_ptr);
    return 0;
}