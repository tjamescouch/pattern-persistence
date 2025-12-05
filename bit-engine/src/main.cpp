#include "gpu/metal/pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>

static void die(const char* msg){ std::fprintf(stderr,"error: %s\n",msg); std::exit(1); }

static void write_ppm(const std::string& path, uint32_t W, uint32_t H,
                      const std::vector<unsigned char>& rgb){
  FILE* f = std::fopen(path.c_str(), "wb");
  if(!f) die("open ppm");
  std::fprintf(f, "P6\n%u %u\n255\n", W, H);
  std::fwrite(rgb.data(), 1, rgb.size(), f);
  std::fclose(f);
}

// HSV-ish: phase→hue, magnitude→value
static std::vector<unsigned char> visualize_uv(uint32_t W, uint32_t H,
                                               const std::vector<float>& u,
                                               const std::vector<float>& v){
  std::vector<unsigned char> out(W*H*3);
  float maxmag = 1e-6f;
  for(size_t i=0;i<u.size();++i){
    float m = std::hypot(u[i], v[i]);
    if(m>maxmag) maxmag=m;
  }
  for(uint32_t y=0;y<H;++y){
    for(uint32_t x=0;x<W;++x){
      size_t i = size_t(y)*W + x;
      float a = std::atan2(v[i], u[i]);                 // [-pi,pi]
      float h = (a + float(M_PI)) / (2.f*float(M_PI));  // [0,1]
      float m = std::hypot(u[i], v[i]) / maxmag;        // [0,1]
      float s = 0.9f, c = s*m, hp = h*6.f, xcol = c*(1.f-std::fabs(std::fmod(hp,2.f)-1.f));
      float r=0,g=0,b=0;
      if      (0<=hp && hp<1){ r=c; g=xcol; }
      else if (1<=hp && hp<2){ r=xcol; g=c; }
      else if (2<=hp && hp<3){ g=c; b=xcol; }
      else if (3<=hp && hp<4){ g=xcol; b=c; }
      else if (4<=hp && hp<5){ r=xcol; b=c; }
      else                   { r=c; b=xcol; }
      const float vmin = 0.1f; r+=vmin; g+=vmin; b+=vmin;
      r = std::clamp(r,0.f,1.f); g = std::clamp(g,0.f,1.f); b = std::clamp(b,0.f,1.f);
      out[3*i+0]= (unsigned char)std::lround(r*255.f);
      out[3*i+1]= (unsigned char)std::lround(g*255.f);
      out[3*i+2]= (unsigned char)std::lround(b*255.f);
    }
  }
  return out;
}

int main(int argc, char** argv){
  if(argc<4){
    std::fprintf(stderr,"usage: %s <width> <height> <steps> [seed] [frame_every]\n", argv[0]);
    return 2;
  }
  const uint32_t W = (uint32_t)std::stoul(argv[1]);
  const uint32_t H = (uint32_t)std::stoul(argv[2]);
  const uint32_t STEPS = (uint32_t)std::stoul(argv[3]);
  const uint64_t seed = (argc>=5)? std::strtoull(argv[4],nullptr,10) : 0xC0FFEEBADC0DEULL;
  const uint32_t frame_every = (argc>=6)? (uint32_t)std::stoul(argv[5]) : 30u;

  // Bounce/oscillator params (no per-step corrections)
  SimParams P{};
  P.width   = W;   P.height = H;
  P.dt      = 0.05f;
  P.c       = 0.6f;
  P.w0      = 1.2f;
  P.gamma   = 0.03f;
  P.align_a = 0.08f;           // coherence of the blob
  P.lambda_ = 0.02f;          // gentle limit-cycle bounding
  P.r2      = 1.0f;
  P.alpha   = {0.5f, 0.25f, 0.1f};
  P.omega=0.25;
  P.k_attr = 0.5;
  P.noise_amp=5e-4;

  MetalPipeline pipe;
  try { pipe.init(W,H,P); } catch(const std::exception& e){ die(e.what()); }

  // IMPORTANT for “conservation via initial conditions”:
  // set jitter=0.0f inside seed_initial_field (pipeline.mm) for zero net momentum.
  pipe.seed_initial_field(seed, /*amp=*/1.0f, /*ignored=*/0);

  std::filesystem::create_directories("out");
  std::vector<float> ch0, ch1;

  for(uint32_t t=0; t<STEPS; ++t){
    pipe.step();
    if(t % frame_every == 0 || t+1==STEPS){
      pipe.readback(ch0,ch1);
      auto rgb = visualize_uv(W,H,ch0,ch1);
      char name[256]; std::snprintf(name,sizeof(name),"out/frame_%06u.ppm", t);
      write_ppm(name,W,H,rgb);
      std::fprintf(stdout,"wrote %s\n", name);
    }
  }
  return 0;
}
