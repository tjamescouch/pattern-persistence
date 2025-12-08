#pragma once
#include <cstdint>
#include <vector>

// Provide Objective-C/Metal refs only to ObjC++ TUs; C++ sees opaque void*.
#ifdef __OBJC__
#import <Metal/Metal.h>
using MTLDeviceRef                = id<MTLDevice>;
using MTLCommandQueueRef          = id<MTLCommandQueue>;
using MTLLibraryRef               = id<MTLLibrary>;
using MTLComputePipelineStateRef  = id<MTLComputePipelineState>;
using MTLBufferRef                = id<MTLBuffer>;
#else
using MTLDeviceRef                = void*;
using MTLCommandQueueRef          = void*;
using MTLLibraryRef               = void*;
using MTLComputePipelineStateRef  = void*;
using MTLBufferRef                = void*;
#endif

class BitPipeline {
public:
  BitPipeline() = default;
  ~BitPipeline();

  // Initialize Metal Device, Command Queue, and load .metallib
  void init();

  // The Core FTT Operation: Float32 -> Int8
  // 1. Uploads src (F32) to GPU
  // 2. Calculates per-row max (Scales)
  // 3. Quantizes to Int8
  // 4. Writes results back to host pointers
  //
  // Arguments:
  // - src:    Pointer to Input Float32 Tensor [rows * dim]
  // - dst:    Pointer to Output Int8 Tensor   [rows * dim]
  // - scales: Pointer to Output Float Scales  [rows]
  // - rows:   Batch size (number of vectors)
  // - dim:    Embedding dimension (e.g., 4096, 8192)
  void dispatch_quantize(const float* src, char* dst, float* scales, uint32_t rows, uint32_t dim);

private:
  MTLDeviceRef               device_{nullptr};
  MTLCommandQueueRef         queue_{nullptr};
  MTLLibraryRef              library_{nullptr};

  // Pipeline State Objects for the kernels defined in universe.metal
  MTLComputePipelineStateRef pso_row_max_{nullptr};
  MTLComputePipelineStateRef pso_quantize_{nullptr};
};