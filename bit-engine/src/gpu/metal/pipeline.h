#pragma once
#include <cstdint>
#include <vector>
#include <array>

// Provide Objective-C/Metal refs only to ObjC++ TUs; C++ sees opaque void*.
#ifdef __OBJC__
#import <Metal/Metal.h>
using MTLDeviceRef                = id<MTLDevice>;
using MTLCommandQueueRef          = id<MTLCommandQueue>;
using MTLLibraryRef               = id<MTLLibrary>;
using MTLComputePipelineStateRef  = id<MTLComputePipelineState>;
using MTLTextureRef               = id<MTLTexture>;
using MTLBufferRef                = id<MTLBuffer>;
#else
using MTLDeviceRef                = void*;
using MTLCommandQueueRef          = void*;
using MTLLibraryRef               = void*;
using MTLComputePipelineStateRef  = void*;
using MTLTextureRef               = void*;
using MTLBufferRef                = void*;
#endif

struct SimParams {
  uint32_t width{0}, height{0};
  float dt{0.05f}, c{0.6f}, w0{1.2f}, gamma{0.06f}, align_a{0.2f};
  float lambda_{0.05f}, r2{1.0f};
  std::array<float,3> alpha{{0.5f,0.25f,0.1f}};
  // NEW:
  float omega{0.15f};
  float k_attr{0.3f};
  float noise_amp{1e-3f};
  uint32_t tstep{0};
};


class MetalPipeline {
public:
  MetalPipeline() = default;
  ~MetalPipeline();

  void init(uint32_t width, uint32_t height, const SimParams& p);
  void seed_initial_field(uint64_t seed, float amp=1.0f, int count=64);
  void step();
  void readback(std::vector<float>& ch0, std::vector<float>& ch1) const;
  void swap_surfaces();

  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

private:
  void create_textures();
  void update_params_buffer();

  uint32_t width_{0}, height_{0};

  MTLDeviceRef               device_{nullptr};
  MTLCommandQueueRef         queue_{nullptr};
  MTLLibraryRef              library_{nullptr};
  MTLComputePipelineStateRef pso_{nullptr};

  MTLTextureRef tex_U_{nullptr};       // texture2d_array<float>, current
  MTLTextureRef tex_U_next_{nullptr};  // next

  MTLTextureRef tex_V_{nullptr};       // texture2d_array<float>, current
  MTLTextureRef tex_V_next_{nullptr};  // next

  MTLBufferRef params_buf_{nullptr};

  SimParams params_{};
};
