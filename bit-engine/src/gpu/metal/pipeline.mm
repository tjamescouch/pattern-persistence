#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "pipeline.h"
#include <stdexcept>
#include <cstring>
#include <random>

static void ensure(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(msg);
}

MetalPipeline::~MetalPipeline() {
  // ARC handles Objective-C objects in ObjC++ builds.
}

void MetalPipeline::init(uint32_t w, uint32_t h, const SimParams& p) {
  width_ = w; height_ = h; params_ = p;

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  ensure(dev != nil, "No Metal device available");
  device_ = dev;

  queue_ = [dev newCommandQueue];
  ensure(queue_ != nil, "Failed to create command queue");

  NSError* err = nil;
  NSURL* url = [NSURL fileURLWithPath:@METALLIB_PATH];
  library_ = [dev newLibraryWithURL:url error:&err];
  ensure(library_ != nil && err == nil, "Failed to load metallib (check METALLIB_PATH)");

  // New kernel name
  id<MTLFunction> fn = [library_ newFunctionWithName:@"wavelet_bounce_step"];
  ensure(fn != nil, "Kernel function 'wavelet_bounce_step' not found");

  pso_ = [dev newComputePipelineStateWithFunction:fn error:&err];
  ensure(pso_ != nil && err == nil, "Failed to create compute pipeline state");

  params_buf_ = [dev newBufferWithLength:sizeof(SimParams) options:MTLResourceStorageModeShared];
  ensure(params_buf_ != nil, "Failed to create params buffer");

  create_textures();
  update_params_buffer();
}

void MetalPipeline::create_textures() {
  // Common descriptor for all ping-pong textures (R32F, arrayLength=2)
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                               width:width_
                                                                              height:height_
                                                                           mipmapped:NO];
  td.textureType = MTLTextureType2DArray;
  td.arrayLength = 2;
  td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

  tex_U_      = [device_ newTextureWithDescriptor:td];
  tex_U_next_ = [device_ newTextureWithDescriptor:td];
  tex_V_      = [device_ newTextureWithDescriptor:td];
  tex_V_next_ = [device_ newTextureWithDescriptor:td];
  ensure(tex_U_ && tex_U_next_ && tex_V_ && tex_V_next_, "Failed to create textures");

  // Zero initialize all
  std::vector<float> zero(width_ * height_, 0.0f);
  MTLRegion r = MTLRegionMake2D(0,0,width_,height_);
  for (int slice = 0; slice < 2; ++slice) {
    [tex_U_      replaceRegion:r mipmapLevel:0 slice:slice withBytes:zero.data() bytesPerRow:width_*sizeof(float) bytesPerImage:zero.size()*sizeof(float)];
    [tex_U_next_ replaceRegion:r mipmapLevel:0 slice:slice withBytes:zero.data() bytesPerRow:width_*sizeof(float) bytesPerImage:zero.size()*sizeof(float)];
    [tex_V_      replaceRegion:r mipmapLevel:0 slice:slice withBytes:zero.data() bytesPerRow:width_*sizeof(float) bytesPerImage:zero.size()*sizeof(float)];
    [tex_V_next_ replaceRegion:r mipmapLevel:0 slice:slice withBytes:zero.data() bytesPerRow:width_*sizeof(float) bytesPerImage:zero.size()*sizeof(float)];
  }
}

void MetalPipeline::update_params_buffer() {
  // Mirror layout in universe.metal::Params
  struct Packed {
    uint32_t width;
    uint32_t height;
    float dt, c, w0, gamma, align_a, lambda_, r2;
    float alpha0, alpha1, alpha2;
  } packed;

  packed.width   = params_.width;
  packed.height  = params_.height;
  packed.dt      = params_.dt;
  packed.c       = params_.c;
  packed.w0      = params_.w0;
  packed.gamma   = params_.gamma;
  packed.align_a = params_.align_a;
  packed.lambda_ = params_.lambda_;
  packed.r2      = params_.r2;
  packed.alpha0  = params_.alpha[0];
  packed.alpha1  = params_.alpha[1];
  packed.alpha2  = params_.alpha[2];

  std::memcpy([params_buf_ contents], &packed, sizeof(packed));
}

void MetalPipeline::seed_initial_field(uint64_t /*seed*/, float amp, int /*count*/) {
  // --- 3-body initial condition: U = sum of radial Gaussians, V = tangential swirl ---
  const float Wf = float(width_), Hf = float(height_);
  const float cx = 0.5f * Wf, cy = 0.5f * Hf;
  const float R  = 0.22f * fmin(Wf, Hf);          // triangle radius from center
  const float sigma = 0.12f * fmin(Wf, Hf);       // blob size (std dev)
  const float inv2s2 = 0.5f / (sigma*sigma);
  const float eps = 1e-6f;

  const int n = 10;

  // Centers at 0°, 120°, 240° around image center
  struct C { float x,y, omega; float sign; };
  C centers[n];
  for (int i=0;i<n;++i) {
    const float th = (float(i) * 2.0f * M_PI) / (n * 1.0f);
    centers[i].x = cx + R * cosf(th);
    centers[i].y = cy + R * sinf(th);
    centers[i].omega = 0.18f + i * 0.05;           // angular speeds (varied)
    centers[i].sign  = (i==2 ? -1.0f : 1.0f);     // alternate spin directions
  }

  std::vector<float> ux(width_*height_, 0.0f);
  std::vector<float> uy(width_*height_, 0.0f);
  std::vector<float> vx(width_*height_, 0.0f);
  std::vector<float> vy(width_*height_, 0.0f);

  for (uint32_t y=0; y<height_; ++y) {
    for (uint32_t x=0; x<width_; ++x) {
      const size_t i = size_t(y)*width_ + x;

      float Ux=0.f, Uy=0.f, Vx=0.f, Vy=0.f;
      for (const auto& c : centers) {
        const float dx = float(x) - c.x;
        const float dy = float(y) - c.y;
        const float r2 = dx*dx + dy*dy;
        const float mag = amp * expf(-r2 * inv2s2);

        // Radial unit vector for U
        const float invr = 1.0f / sqrtf(r2 + eps);
        const float rx = dx * invr, ry = dy * invr;
        Ux += mag * rx;
        Uy += mag * ry;

        // Tangential (perpendicular) for V; magnitude ~ omega * radius * mag
        const float tx = -ry, ty = rx;                 // rotate +90°
        const float swirl = c.sign * c.omega * sqrtf(r2 + eps);
        Vx += swirl * mag * tx;
        Vy += swirl * mag * ty;
      }
      ux[i] = Ux; uy[i] = Uy;
      vx[i] = Vx; vy[i] = Vy;
    }
  }

  // --- Zero DC to avoid drift (conservation via initial conditions) ---
  auto zero_mean = [&](std::vector<float>& a){
    double s=0.0; for (float v : a) s += v;
    const float m = float(s / double(a.size()));
    for (auto& v : a) v -= m;
  };
  zero_mean(ux); zero_mean(uy);
  zero_mean(vx); zero_mean(vy);

  // Optional: normalize overall RMS of U for comparable brightness
  auto rms = [&](const std::vector<float>& a){
    long double s=0.0; for (float v : a) s += (long double)v*v;
    return float(std::sqrt(double(s / a.size())));
  };
  const float target_rms = 0.8f; // tweak if needed
  const float cur = 0.5f * (rms(ux)+rms(uy)) + 1e-6f;
  const float scale = target_rms / cur;
  for (size_t i=0;i<ux.size();++i){ ux[i]*=scale; uy[i]*=scale; }
  // Scale V proportionally so initial CFL stays sensible
  for (size_t i=0;i<vx.size();++i){ vx[i]*=0.6f*scale; vy[i]*=0.6f*scale; }

  // Upload to Metal (U slices 0,1; V slices 0,1)
  MTLRegion r = MTLRegionMake2D(0,0,width_,height_);
  [tex_U_ replaceRegion:r mipmapLevel:0 slice:0 withBytes:ux.data()
            bytesPerRow:width_*sizeof(float) bytesPerImage:ux.size()*sizeof(float)];
  [tex_U_ replaceRegion:r mipmapLevel:0 slice:1 withBytes:uy.data()
            bytesPerRow:width_*sizeof(float) bytesPerImage:uy.size()*sizeof(float)];
  [tex_V_ replaceRegion:r mipmapLevel:0 slice:0 withBytes:vx.data()
            bytesPerRow:width_*sizeof(float) bytesPerImage:vx.size()*sizeof(float)];
  [tex_V_ replaceRegion:r mipmapLevel:0 slice:1 withBytes:vy.data()
            bytesPerRow:width_*sizeof(float) bytesPerImage:vy.size()*sizeof(float)];
}



void MetalPipeline::swap_surfaces() {
  auto swapTex = [](id<MTLTexture>& a, id<MTLTexture>& b){ id<MTLTexture> t=a; a=b; b=t; };
  swapTex(tex_U_, tex_U_next_);
  swapTex(tex_V_, tex_V_next_);
}

void MetalPipeline::step() {
  update_params_buffer();

  id<MTLCommandBuffer> cb = [(__bridge id<MTLCommandQueue>)queue_ commandBuffer];
  ensure(cb != nil, "Failed to create command buffer");

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  ensure(enc != nil, "Failed to create compute encoder");

  [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso_];

  // Bind in the order required by the new kernel:
  // 0: U_t (read), 1: V_t (read), 2: U_next (write), 3: V_next (write)
  [enc setTexture:(__bridge id<MTLTexture>)tex_U_      atIndex:0];
  [enc setTexture:(__bridge id<MTLTexture>)tex_V_      atIndex:1];
  [enc setTexture:(__bridge id<MTLTexture>)tex_U_next_ atIndex:2];
  [enc setTexture:(__bridge id<MTLTexture>)tex_V_next_ atIndex:3];

  [enc setBuffer:(__bridge id<MTLBuffer>)params_buf_ offset:0 atIndex:0];

  MTLSize tg   = MTLSizeMake(16,16,1);
  MTLSize grid = MTLSizeMake(
      ((width_  + tg.width  - 1) / tg.width ) * tg.width,
      ((height_ + tg.height - 1) / tg.height) * tg.height,
      1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  [enc endEncoding];

  [cb commit];
  [cb waitUntilCompleted];

  swap_surfaces();
}

void MetalPipeline::readback(std::vector<float>& ch0, std::vector<float>& ch1) const {
  ch0.resize(width_*height_);
  ch1.resize(width_*height_);
  MTLRegion r = MTLRegionMake2D(0,0,width_,height_);
  [(__bridge id<MTLTexture>)tex_U_ getBytes:ch0.data() bytesPerRow:width_*sizeof(float)
                          bytesPerImage:ch0.size()*sizeof(float)
                                fromRegion:r mipmapLevel:0 slice:0];
  [(__bridge id<MTLTexture>)tex_U_ getBytes:ch1.data() bytesPerRow:width_*sizeof(float)
                          bytesPerImage:ch1.size()*sizeof(float)
                                fromRegion:r mipmapLevel:0 slice:1];
}
