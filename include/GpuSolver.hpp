#ifndef GPU_SOLVER_HPP
#define GPU_SOLVER_HPP

#include "GpuKernelCore.cuh"
#include <fstream>

namespace KG {

class GpuSolver {
public:
    GpuSolver(GpuLattice& lattice);
    ~GpuSolver();

    void initialize_gaussian_clusters();
    void initialize_idle_proton();
    void step(double dt);
    
    // Observation
    void record_boundary_flux(int step);

private:
    GpuLattice& lattice_;
    cudaStream_t physics_stream_;
    cudaStream_t observer_stream_;
    
    std::ofstream monitor_log_;
    
    // Domain Decomposition (Prototype for multi-GPU)
    int offset_x, offset_y, offset_z;
    int local_nx, local_ny, local_nz;

    void launch_weno_advection(double dt);
    void launch_density_gate();
};

} // namespace KG

#endif // GPU_SOLVER_HPP
