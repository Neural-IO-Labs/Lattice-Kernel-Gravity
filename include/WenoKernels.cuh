#ifndef WENO_KERNELS_CUH
#define WENO_KERNELS_CUH

#include "GpuKernelCore.cuh"

namespace KG {

// 5th-order WENO Reconstruction Kernels
__global__ void weno5_advection_x(GpuLatticeData data, GpuLatticeData buffer, double dt);
__global__ void weno5_advection_y(GpuLatticeData data, GpuLatticeData buffer, double dt);
__global__ void weno5_advection_z(GpuLatticeData data, GpuLatticeData buffer, double dt);

// The Density Gate / Planck Limit Logic
__global__ void planck_limit_gate(GpuLatticeData data, GpuLatticeData buffer);

// Initialization helper
__global__ void init_gaussian_cluster(GpuLatticeData data, double cx, double cy, double cz, double mass, double vx);

// Reduction Kernels for Observation
__global__ void reduce_stats(GpuLatticeData data, double* out_mass, double* out_clock_sum);

} // namespace KG

#endif // WENO_KERNELS_CUH
