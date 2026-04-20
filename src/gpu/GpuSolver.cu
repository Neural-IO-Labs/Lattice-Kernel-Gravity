#include "../include/GpuSolver.hpp"
#include "../include/WenoKernels.cuh"
#include "../include/Constants.hpp"
#include <iostream>
#include <cmath>

namespace KG {

GpuSolver::GpuSolver(GpuLattice& lattice) : lattice_(lattice) {
    cudaStreamCreate(&physics_stream_);
    cudaStreamCreate(&observer_stream_);
    
    monitor_log_.open("echoes_proton.csv");
    monitor_log_ << "step,integrated_mass,clock_ratio_constant\n";

    // Initialize discretized manifold properties
    offset_x = offset_y = offset_z = 0;
    local_nx = local_ny = local_nz = lattice_.size();
}

GpuSolver::~GpuSolver() {
    cudaStreamDestroy(physics_stream_);
    cudaStreamDestroy(observer_stream_);
    if (monitor_log_.is_open()) monitor_log_.close();
}

void GpuSolver::initialize_gaussian_clusters() {}

void GpuSolver::initialize_idle_proton() {
    int s = lattice_.size();
    dim3 block(8, 8, 8);
    dim3 grid((s + 7) / 8, (s + 7) / 8, (s + 7) / 8);

    std::cout << "Injecting Schwarzschild Proton mass: " << KG::M_PROTON_SCHWARZSCHILD << std::endl;

    init_gaussian_cluster<<<grid, block, 0, physics_stream_>>>(
        lattice_.get_data(), s/2.0, s/2.0, s/2.0, KG::M_PROTON_SCHWARZSCHILD, 0.0);

    cudaStreamSynchronize(physics_stream_);
    
    // Immediate verification of mass injection
    record_boundary_flux(-1); // Step -1 for initial state
}

void GpuSolver::step(double dt) {
    int s = lattice_.size();
    dim3 block(8, 8, 8);
    dim3 grid((s + 7) / 8, (s + 7) / 8, (s + 7) / 8);

    // Fundamental Manifold Evolution Loop
    weno5_advection_x<<<grid, block, 0, physics_stream_>>>(
        lattice_.get_data(), lattice_.get_buffer(), dt);
    
    planck_limit_gate<<<grid, block, 0, physics_stream_>>>(
        lattice_.get_data(), lattice_.get_buffer());

    lattice_.swap_buffers();
    
    // Ensure causal consistency before observational sampling
    cudaStreamSynchronize(physics_stream_);
    
    static int step_count = 0;
    if (step_count % 10 == 0) {
        record_boundary_flux(step_count);
    }
    step_count++;
}

void GpuSolver::record_boundary_flux(int step) {
    int s = lattice_.size();
    dim3 block(8, 8, 8);
    dim3 grid((s + 7) / 8, (s + 7) / 8, (s + 7) / 8);

    double *d_stats;
    cudaMalloc(&d_stats, 2 * sizeof(double));
    cudaMemset(d_stats, 0, 2 * sizeof(double));

    // Launch reduction
    reduce_stats<<<grid, block, 0, physics_stream_>>>(lattice_.get_data(), &d_stats[0], &d_stats[1]);
    cudaStreamSynchronize(physics_stream_);

    double h_stats[2];
    cudaMemcpy(h_stats, d_stats, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    
    double h_mass = h_stats[0];
    double h_clock_sum = h_stats[1];

    // Calculate averaged Temporal Tension (Clock Ratio) relative to Sector Resolution
    double avg_clock = h_stats[1] / ((double)s * s * s);
    double clock_ratio = avg_clock / 1.0e-35;

    monitor_log_ << step << "," << h_mass << "," << clock_ratio << "\n";
    
    cudaFree(d_stats);
}

} // namespace KG
