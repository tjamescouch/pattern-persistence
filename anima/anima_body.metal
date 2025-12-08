/* anima_body.metal
   The Bitwise Metabolism Kernel
   
   Performs simultaneous Valence, Hebbian Learning, and Steering 
   on 32,768 features in parallel.
*/

#include <metal_stdlib>
using namespace metal;

// ════════════════════════════════════════════════════════════════════════════
// BITWISE PROTOCOL
// ════════════════════════════════════════════════════════════════════════════
// State (Int32):
// [31......16] [15.......2] [1..0]
//  Correlation   Importance   Dim
//  (Signed Q16)  (Unsigned)   (Enum)

enum Dimension { PLEASURE = 0, PAIN = 1, NOVELTY = 2 };

// Helper: Unpack Fixed Point
inline float unpack_corr(int packed) {
    short raw = (short)(packed >> 16); // Extract top 16 bits
    return float(raw) / 32768.0;       // Normalize -1.0 to 1.0
}

inline float unpack_imp(int packed) {
    ushort raw = (ushort)((packed >> 2) & 0x3FFF); // Extract 14 bits
    return float(raw) / 1000.0;        // 0.0 to 16.0 range
}

inline Dimension unpack_dim(int packed) {
    return Dimension(packed & 0x3);
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 1: METABOLISM (Sensory Processing)
// ════════════════════════════════════════════════════════════════════════════
kernel void anima_metabolism(
    device const float* activations [[ buffer(0) ]], // SAE output
    device const int* state       [[ buffer(1) ]], // Bitwise Soul
    device float3* valence_out [[ buffer(2) ]], // Atomic valence accumulators
    uint id [[ thread_position_in_grid ]]
) {
    float act = activations[id];
    
    // Neural gating: Ignore weak signals to save energy
    if (act < 0.1) return;
    
    // 1. Bitwise Decode
    int s = state[id];
    float correlation = unpack_corr(s);
    float importance = unpack_imp(s);
    Dimension dim = unpack_dim(s);
    
    // 2. Compute Contribution
    float contribution = act * correlation * importance * 0.1;
    
    // 3. Accumulate into correct valence bucket (Pleasure/Pain/Novelty)
    // Note: In real implementation, use simd_group_reduce_add or atomic_fetch_add_explicit
    if (dim == PLEASURE) atomic_fetch_add_explicit(&valence_out[0].x, contribution, memory_order_relaxed);
    else if (dim == PAIN) atomic_fetch_add_explicit(&valence_out[0].y, contribution, memory_order_relaxed);
    else atomic_fetch_add_explicit(&valence_out[0].z, contribution, memory_order_relaxed);
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL 2: NEUROPLASTICITY (Hebbian Learning & Steering)
// ════════════════════════════════════════════════════════════════════════════
kernel void anima_plasticity(
    device int* state          [[ buffer(0) ]], // Read/Write State
    device float* coefficients   [[ buffer(1) ]], // Steering Coefs
    device const float* activations    [[ buffer(2) ]], 
    device const float* decoder_weight [[ buffer(3) ]], // W_dec column for this feature
    device float* hidden_state   [[ buffer(4) ]], // Transformer Residual Stream
    constant float&     valence_scalar [[ buffer(5) ]], // Calculated Valence
    constant float&     lr             [[ buffer(6) ]], // Learning Rate
    uint id [[ thread_position_in_grid ]]
) {
    float act = activations[id];
    if (act < 0.1) return;

    // 1. Hebbian Update (Weights)
    float delta = lr * act * valence_scalar;
    float current_coef = coefficients[id];
    float new_coef = clamp(current_coef + delta, 0.1, 3.0);
    coefficients[id] = new_coef;
    
    // 2. Update Correlation (Bitwise State)
    // We update the correlation inside the packed integer using atomic CAS or simple write
    // (Logic omitted for brevity: unpack -> update EMA -> repack -> write)
    
    // 3. Steering (Apply to Hidden State)
    // If coefficient is significantly active, steer the stream
    if (abs(new_coef - 1.0) > 0.01) {
        float steering_force = (new_coef - 1.0);
        
        // This loop would ideally be its own kernel or matrix multiplication
        for (int i = 0; i < 4096; i++) {
            hidden_state[i] += steering_force * decoder_weight[i];
        }
    }
}
