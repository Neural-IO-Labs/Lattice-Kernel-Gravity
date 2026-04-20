#ifndef GPU_KERNEL_CORE_CUH
#define GPU_KERNEL_CORE_CUH

#include <cuda_runtime.h>
#include "Constants.hpp"

namespace KG {

// SoA (Structure of Arrays) for optimal GPU coalescing
struct GpuLatticeData {
    double* density;
    double* vx;
    double* vy;
    double* vz;
    double* clock_speed;
    int size;
};

class GpuLattice {
public:
    GpuLattice(int size);
    ~GpuLattice();

    void allocate();
    void deallocate();
    
    // Transfers data to/from host for setup/monitoring
    void copy_to_host(double* h_density, double* h_clock);
    void copy_from_host(const double* h_density);

    GpuLatticeData get_data() { return data_; }
    GpuLatticeData get_buffer() { return buffer_; }
    
    void swap_buffers();
    int size() const { return size_; }

private:
    int size_;
    size_t num_cells_;
    GpuLatticeData data_;
    GpuLatticeData buffer_;
};

} // namespace KG

#endif // GPU_KERNEL_CORE_CUH
