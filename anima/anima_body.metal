/* anima_body.metal - Anima v9.3 Body Kernel */
#include <metal_stdlib>
using namespace metal;

// ════════════════════════════════════════════════════════════════════════════
// PROTOCOL: Int32 State
// [31......16] [15.......2] [1..0]
//  Correlation   Importance   Dim
//  (Signed Q16)  (Unsigned)   (Enum)
// ════════════════════════════════════════════════════════════════════════════

enum Dimension { PLEASURE = 0, PAIN = 1, NOVELTY = 2 };

inline float unpack_corr(int packed) {
    short raw = (short)(packed >> 16); 
    return float(raw) / 32768.0;       
}

inline Dimension unpack_dim(int packed) {
    return Dimension(packed & 0x3);
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL: METABOLISM
// ════════════════════════════════════════════════════════════════════════════
kernel void anima_metabolism(
    device const float* activations [[ buffer(0) ]], 
    device const int* state         [[ buffer(1) ]], 
    device atomic_float* valence_out [[ buffer(2) ]], 
    uint id [[ thread_position_in_grid ]]
) {
    // Safety check for 32k features
    if (id >= 32768) return;

    float act = activations[id];
    
    // Neural Gating: Ignore weak signals
    if (act < 0.1) return;
    
    // Decode Memory
    int s = state[id];
    float correlation = unpack_corr(s);
    
    // Ignore neutral features
    if (abs(correlation) < 0.01) return;

    // Calculate intensity
    float contribution = act * correlation * 0.1;
    Dimension dim = unpack_dim(s);
    
    // Atomic Accumulation
    if (dim == PLEASURE) {
        atomic_fetch_add_explicit(&valence_out[0], contribution, memory_order_relaxed);
    } 
    else if (dim == PAIN) {
        atomic_fetch_add_explicit(&valence_out[1], contribution, memory_order_relaxed);
    } 
    else {
        atomic_fetch_add_explicit(&valence_out[2], contribution, memory_order_relaxed);
    }
}