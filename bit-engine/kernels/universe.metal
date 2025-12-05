#include <metal_stdlib>
using namespace metal;

/* Second-order lattice oscillator with swirl + nonlocal attraction + tiny noise */

struct Params {
  uint  width, height;
  float dt, c, w0, gamma, align_a, lambda_, r2;
  float alpha0, alpha1, alpha2;
  // NEW:
  float omega;      // Coriolis/rotation strength (≈0..0.5)
  float k_attr;     // attraction to ∇(LP |U|^2), >0 clusters, <0 disperses
  float noise_amp;  // tiny deterministic drive (≈1e-3)
  uint  tstep;      // integer timestep for deterministic noise
};

inline uint2 wrap(uint2 p, uint W, uint H){ return uint2((p.x+W)%W, (p.y+H)%H); }
inline float sread(texture2d_array<float, access::read> tex, uint2 p, uint slice){
  uint2 q = wrap(p, tex.get_width(), tex.get_height()); return tex.read(q, slice).r;
}

// 5-tap kernels
constant float T5[5]={-1.0f/16,-0.25f,0,0.25f,1.0f/16};
constant float S5[5]={ 1.0f/16, 0.25f,3.0f/8,0.25f,1.0f/16};

inline float sep_band(texture2d_array<float, access::read> tex, uint2 p, uint s, uint stride){
  float h=0, v=0, hs=0, vs=0;
  for(int k=-2;k<=2;++k){
    h  += T5[k+2]*sread(tex,uint2(p.x+k*stride,p.y),s);
    v  += T5[k+2]*sread(tex,uint2(p.x,p.y+k*stride),s);
    hs += S5[k+2]*sread(tex,uint2(p.x+k*stride,p.y),s);
    vs += S5[k+2]*sread(tex,uint2(p.x,p.y+k*stride),s);
  }
  return 0.5f*(h+v) - 0.5f*(hs+vs);
}
inline float sep_lowpass(texture2d_array<float, access::read> tex, uint2 p, uint s){
  float h=0, v=0; for(int k=-2;k<=2;++k){ h+=S5[k+2]*sread(tex,uint2(p.x+k,p.y),s);
                                          v+=S5[k+2]*sread(tex,uint2(p.x,p.y+k),s); }
  return 0.5f*(h+v);
}

// Tiny, deterministic hash -> [-1,1]
inline float dither(uint2 p, uint t){
  uint n = p.x*374761393u + p.y*668265263u + t*362437u; // LCG mix
  n = (n ^ (n>>13u)) * 1274126177u;
  return ((float)(n & 0x00FFFFFFu) / 8388608.0f) - 1.0f; // ~[-1,1]
}

kernel void wavelet_bounce_step(
  texture2d_array<float, access::read>   U_t     [[texture(0)]],
  texture2d_array<float, access::read>   V_t     [[texture(1)]],
  texture2d_array<float, access::write>  U_next  [[texture(2)]],
  texture2d_array<float, access::write>  V_next  [[texture(3)]],
  constant Params& P                               [[buffer(0)]],
  uint2 gid                                        [[thread_position_in_grid]])
{
  if (gid.x>=P.width || gid.y>=P.height) return;

  // State
  float ux = sread(U_t,gid,0), uy = sread(U_t,gid,1);
  float vx = sread(V_t,gid,0), vy = sread(V_t,gid,1);

  // Dispersive operator L[U]
  float Lx = P.alpha0*sep_band(U_t,gid,0,1) + P.alpha1*sep_band(U_t,gid,0,2) + P.alpha2*sep_band(U_t,gid,0,4);
  float Ly = P.alpha0*sep_band(U_t,gid,1,1) + P.alpha1*sep_band(U_t,gid,1,2) + P.alpha2*sep_band(U_t,gid,1,4);

  // Alignment
  float ux_lp = sep_lowpass(U_t, gid, 0);
  float uy_lp = sep_lowpass(U_t, gid, 1);

  // Limit-cycle bounding
  float u2 = ux*ux + uy*uy;
  float cyc = (P.r2>0.0f)? (1.0f - u2/P.r2) : 0.0f;
  float fx  = P.lambda_ * ux * cyc;
  float fy  = P.lambda_ * uy * cyc;

  // --- NEW: nonlocal attraction/repulsion via ∇(LP |U|^2) ---
  float e_lp = sep_lowpass(U_t, gid, 0)*sep_lowpass(U_t, gid, 0)
             + sep_lowpass(U_t, gid, 1)*sep_lowpass(U_t, gid, 1);
  // central differences on the low-pass energy
  float ex1 = sep_lowpass(U_t, wrap(uint2(gid.x+1,gid.y), P.width, P.height),0);
  float ex2 = sep_lowpass(U_t, wrap(uint2(gid.x-1,gid.y), P.width, P.height),0);
  float ey1 = sep_lowpass(U_t, wrap(uint2(gid.x,gid.y+1), P.width, P.height),0);
  float ey2 = sep_lowpass(U_t, wrap(uint2(gid.x,gid.y-1), P.width, P.height),0);
  float fx1 = sep_lowpass(U_t, wrap(uint2(gid.x+1,gid.y), P.width, P.height),1);
  float fx2 = sep_lowpass(U_t, wrap(uint2(gid.x-1,gid.y), P.width, P.height),1);
  float fy1 = sep_lowpass(U_t, wrap(uint2(gid.x,gid.y+1), P.width, P.height),1);
  float fy2 = sep_lowpass(U_t, wrap(uint2(gid.x,gid.y-1), P.width, P.height),1);
  float dEx = (ex1*ex1+fx1*fx1) - (ex2*ex2+fx2*fx2);
  float dEy = (ey1*ey1+fy1*fy1) - (ey2*ey2+fy2*fy2);
  float attrx = -0.5f * P.k_attr * dEx;
  float attry = -0.5f * P.k_attr * dEy;

  // --- NEW: Coriolis/rotation (precession of velocity) ---
  float cori_x =  P.omega * (-vy);
  float cori_y =  P.omega * ( vx);

  // --- NEW: tiny deterministic, band-limited noise drive ---
  float n0 = dither(gid, P.tstep);
  float n1 = dither(gid^uint2(911u,613u), P.tstep);
  // low-pass it by mixing with alignment low-pass
  n0 = 0.2f*n0 + 0.8f*sep_lowpass(U_t, gid, 0);
  n1 = 0.2f*n1 + 0.8f*sep_lowpass(U_t, gid, 1);

  // Acceleration
  float ax = (P.c*P.c)*Lx - (P.w0*P.w0)*ux - P.gamma*vx + P.align_a*(ux_lp-ux) + fx
             + attrx + cori_x + P.noise_amp*n0;
  float ay = (P.c*P.c)*Ly - (P.w0*P.w0)*uy - P.gamma*vy + P.align_a*(uy_lp-uy) + fy
             + attry + cori_y + P.noise_amp*n1;

  // Symplectic Euler
  float vx1 = vx + P.dt*ax, vy1 = vy + P.dt*ay;
  float ux1 = ux + P.dt*vx1, uy1 = uy + P.dt*vy1;

  U_next.write(ux1,gid,0); U_next.write(uy1,gid,1);
  V_next.write(vx1,gid,0); V_next.write(vy1,gid,1);
}
