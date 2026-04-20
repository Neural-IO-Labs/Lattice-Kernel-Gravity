#include "../include/GpuKernelCore.cuh"
#include <iostream>

namespace KG {

GpuLattice::GpuLattice(int size) : size_(size) {
    num_cells_ = (size_t)size_ * size_ * size_;
    allocate();
}

GpuLattice::~GpuLattice() {
    deallocate();
}

void GpuLattice::allocate() {
    size_t d_bytes = num_cells_ * sizeof(double);
    
    // Primary Buffer
    cudaMalloc(&data_.density, d_bytes);
    cudaMalloc(&data_.vx, d_bytes);
    cudaMalloc(&data_.vy, d_bytes);
    cudaMalloc(&data_.vz, d_bytes);
    cudaMalloc(&data_.clock_speed, d_bytes);
    data_.size = size_;

    // Secondary Buffer (Double Buffering)
    cudaMalloc(&buffer_.density, d_bytes);
    cudaMalloc(&buffer_.vx, d_bytes);
    cudaMalloc(&buffer_.vy, d_bytes);
    cudaMalloc(&buffer_.vz, d_bytes);
    cudaMalloc(&buffer_.clock_speed, d_bytes);
    buffer_.size = size_;

    // Initialize to zero
    cudaMemset(data_.density, 0, d_bytes);
    cudaMemset(data_.vx, 0, d_bytes);
    cudaMemset(data_.vy, 0, d_bytes);
    cudaMemset(data_.vz, 0, d_bytes);
    cudaMemset(data_.clock_speed, 0, d_bytes); // Set to 1.0 later in setup
}

void GpuLattice::deallocate() {
    cudaFree(data_.density);
    cudaFree(data_.vx);
    cudaFree(data_.vy);
    cudaFree(data_.vz);
    cudaFree(data_.clock_speed);

    cudaFree(buffer_.density);
    cudaFree(buffer_.vx);
    cudaFree(buffer_.vy);
    cudaFree(buffer_.vz);
    cudaFree(buffer_.clock_speed);
}

void GpuLattice::swap_buffers() {
    std::swap(data_.density, buffer_.density);
    std::swap(data_.vx, buffer_.vx);
    std::swap(data_.vy, buffer_.vy);
    std::swap(data_.vz, buffer_.vz);
    std::swap(data_.clock_speed, buffer_.clock_speed);
}

void GpuLattice::copy_to_host(double* h_density, double* h_clock) {
    size_t d_bytes = num_cells_ * sizeof(double);
    if (h_density) cudaMemcpy(h_density, data_.density, d_bytes, cudaMemcpyDeviceToHost);
    if (h_clock) cudaMemcpy(h_clock, data_.clock_speed, d_bytes, cudaMemcpyDeviceToHost);
}

void GpuLattice::copy_from_host(const double* h_density) {
    size_t d_bytes = num_cells_ * sizeof(double);
    cudaMemcpy(data_.density, h_density, d_bytes, cudaMemcpyHostToDevice);
}

} // namespace KG
