#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "pipeline.h"
#include <stdexcept>
#include <cstring>
#include <iostream>

static void ensure(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "[Metal Error] " << msg << std::endl;
    throw std::runtime_error(msg);
  }
}

BitPipeline::~BitPipeline() {
  // ARC handles release in ObjC++ mode
}

void BitPipeline::init() {
  // 1. Get Device
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  ensure(dev != nil, "No Metal device available");
  device_ = dev;

  // 2. Command Queue
  queue_ = [dev newCommandQueue];
  ensure(queue_ != nil, "Failed to create command queue");

  // 3. Load Library (.metallib)
  NSError* err = nil;
  // We assume the metallib is next to the executable or defined by macro
#ifdef METALLIB_PATH
  NSURL* url = [NSURL fileURLWithPath:@METALLIB_PATH];
  library_ = [dev newLibraryWithURL:url error:&err];
#else
  library_ = [dev newDefaultLibrary]; // Fallback to embedded default
#endif
  
  if (library_ == nil) {
      std::cerr << "Err: " << [[err localizedDescription] UTF8String] << std::endl;
  }
  ensure(library_ != nil, "Failed to load metallib");

  // 4. Load Kernels
  auto load_kernel = [&](const char* name) -> id<MTLComputePipelineState> {
    NSString* ns_name = [NSString stringWithUTF8String:name];
    id<MTLFunction> fn = [library_ newFunctionWithName:ns_name];
    if (fn == nil) return nil;
    
    return [device_ newComputePipelineStateWithFunction:fn error:&err];
  };

  pso_row_max_ = load_kernel("find_row_max");
  ensure(pso_row_max_ != nil, "Kernel 'find_row_max' missing");

  pso_quantize_ = load_kernel("quantize_f32_int8");
  ensure(pso_quantize_ != nil, "Kernel 'quantize_f32_int8' missing");
  
  std::cout << "[-] Metal Bit-Engine Initialized." << std::endl;
}

void BitPipeline::dispatch_quantize(const float* src, char* dst, float* scales, uint32_t rows, uint32_t dim) {
  if (!device_ || !src || !dst || !scales) return;

  // --- MEMORY MANAGEMENT ---
  // For the "First Execution Step", we use explicit copies to avoid alignment crashes.
  // Optimization V2: Use newBufferWithBytesNoCopy (Requires page-aligned host memory).
  
  // 1. Create Buffers
  NSUInteger src_bytes = rows * dim * sizeof(float);
  NSUInteger dst_bytes = rows * dim * sizeof(char);
  NSUInteger scl_bytes = rows * sizeof(float);

  // Input: Copy host -> GPU
  id<MTLBuffer> buf_src = [device_ newBufferWithBytes:src 
                                               length:src_bytes 
                                              options:MTLResourceStorageModeShared];
  
  // Output: Alloc on GPU (Shared)
  id<MTLBuffer> buf_dst = [device_ newBufferWithLength:dst_bytes 
                                               options:MTLResourceStorageModeShared];
                                               
  id<MTLBuffer> buf_scl = [device_ newBufferWithLength:scl_bytes 
                                               options:MTLResourceStorageModeShared];
                                               
  // Constant for Dimensions
  id<MTLBuffer> buf_dim = [device_ newBufferWithBytes:&dim 
                                               length:sizeof(uint32_t) 
                                              options:MTLResourceStorageModeShared];

  // --- COMMAND ENCODING ---
  id<MTLCommandBuffer> cb = [queue_ commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

  // PASS 1: Find Row Max (Determine Scales)
  // Grid: 1 Threadgroup per Row
  // Threads: 256 per group (or less if dim is small)
  {
      [enc setComputePipelineState:pso_row_max_];
      [enc setBuffer:buf_src offset:0 atIndex:0];
      [enc setBuffer:buf_scl offset:0 atIndex:1];
      [enc setBuffer:buf_dim offset:0 atIndex:2];

      NSUInteger threadsPerGroup = 256; 
      // Ensure we don't exceed max threads per group for the device, usually 1024
      if (threadsPerGroup > pso_row_max_.maxTotalThreadsPerThreadgroup) {
          threadsPerGroup = pso_row_max_.maxTotalThreadsPerThreadgroup;
      }
      
      MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
      MTLSize grid = MTLSizeMake(rows, 1, 1); // One group per row
      
      // Note: dispatchThreadgroups requires non-uniform threadgroup support (Apple Silicon has this)
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgSize];
  }
  
  // Memory Barrier: Ensure scales are written before read by quantize?
  // Within same encoder, Metal tracks dependencies via buffer usage, but strictly safely we barrier or split encoders.
  // Implicit barrier usually sufficient for straight producer-consumer in same buffer, but let's be safe.
  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

  // PASS 2: Quantize
  // Grid: Flat 1D over all elements
  {
      [enc setComputePipelineState:pso_quantize_];
      [enc setBuffer:buf_src offset:0 atIndex:0];
      [enc setBuffer:buf_scl offset:0 atIndex:1];
      [enc setBuffer:buf_dst offset:0 atIndex:2];
      [enc setBuffer:buf_dim offset:0 atIndex:3]; // Re-bind dim if needed

      NSUInteger total_elements = rows * dim;
      NSUInteger threadsPerGroup = 256;
      MTLSize tgSize = MTLSizeMake(threadsPerGroup, 1, 1);
      MTLSize grid = MTLSizeMake((total_elements + threadsPerGroup - 1) / threadsPerGroup, 1, 1);
      
      // Use dispatchThreads for auto-bounds checking (if supported) or threadgroups
      // Using standard grid dispatch:
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgSize];
  }

  [enc endEncoding];
  [cb commit];
  
  // --- SYNC & READBACK ---
  [cb waitUntilCompleted];

  // Copy GPU results back to Host Pointers
  memcpy(dst, [buf_dst contents], dst_bytes);
  memcpy(scales, [buf_scl contents], scl_bytes);
}