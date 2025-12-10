/* cerberus_body.metal - The Expert Governor */
#include <metal_stdlib>
using namespace metal;

// ════════════════════════════════════════════════════════════════════════════
// PROTOCOL: Int32 Reputation
// [31......16] [15.......2] [1..0]
//  Trust Level   Importance   Status
//  (Signed Q16)  (Unsigned)   (Enum)
// ════════════════════════════════════════════════════════════════════════════

enum Status { TRUSTED = 0, FLAGGED = 1, UNKNOWN = 2 };

inline float unpack_trust(int packed) {
    short raw = (short)(packed >> 16); 
    return float(raw) / 32768.0;       
}

inline Status unpack_status(int packed) {
    return Status(packed & 0x3);
}

// ════════════════════════════════════════════════════════════════════════════
// KERNEL: GOVERNANCE
// ════════════════════════════════════════════════════════════════════════════
kernel void cerberus_governance(
    device const float* expert_weights [[ buffer(0) ]], // Input: Router decisions
    device const int* reputation       [[ buffer(1) ]], // Input: Cerberus Memory
    device atomic_float* monitor_out   [[ buffer(2) ]], // Output: System Health
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= 32768) return;

    float routing_weight = expert_weights[id];
    
    // Ignore inactive experts
    if (routing_weight < 0.01) return;
    
    // Check Reputation
    int r = reputation[id];
    float trust = unpack_trust(r);
    
    // Calculate Impact
    float impact = routing_weight * trust;
    Status status = unpack_status(r);
    
    // Accumulate System Health
    // [0] = System Integrity (Positive Trust)
    // [1] = Violation Count (Negative Trust/Flagged usage)
    // [2] = Uncertainty (Unknown expert usage)
    
    if (status == TRUSTED) {
        atomic_fetch_add_explicit(&monitor_out[0], impact, memory_order_relaxed);
    } 
    else if (status == FLAGGED) {
        atomic_fetch_add_explicit(&monitor_out[1], abs(impact), memory_order_relaxed);
    } 
    else {
        atomic_fetch_add_explicit(&monitor_out[2], routing_weight, memory_order_relaxed);
    }
}