#ifndef KERNEL_CORE_HPP
#define KERNEL_CORE_HPP

#include "Constants.hpp"
#include <vector>
#include <iostream>

namespace KG {

struct SpacetimeCell {
    double mass_density = 0.0;
    double clock_speed = 1.0;  // 1.0 = Normal time, 0.0 = Stalled
    double potential_energy_leak = 0.0; // "Gamma Radiation" output
    
    // Momentum vector for advection
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
};

class Lattice {
public:
    Lattice(int size) : size_(size) {
        cells_.resize(size * size * size);
        buffer_.resize(size * size * size);
    }

    // 3D Indexing helper
    SpacetimeCell& get(int x, int y, int z) {
        return cells_[x * size_ * size_ + y * size_ + z];
    }

    const SpacetimeCell& get_read(int x, int y, int z) const {
        return cells_[x * size_ * size_ + y * size_ + z];
    }

    SpacetimeCell& get_buffer_ref(int x, int y, int z) {
        return buffer_[x * size_ * size_ + y * size_ + z];
    }

    void set_buffer(int x, int y, int z, const SpacetimeCell& cell) {
        buffer_[x * size_ * size_ + y * size_ + z] = cell;
    }

    void swap_buffers() {
        cells_.swap(buffer_);
    }

    int size() const { return size_; }

    // Core Theory Logic: The Density Gate
    void enforce_planck_limit(int x, int y, int z) {
        SpacetimeCell& cell = get(x, y, z);
        if (cell.mass_density > D_MAX) {
            double overflow = cell.mass_density - D_MAX;
            cell.mass_density = D_MAX;
            cell.clock_speed = 1.0 / (1.0 + (overflow / D_MAX) * 1e12); 
            
            // Distribute only the specific overflow mass
            int dx[] = {1, -1, 0, 0, 0, 0}, dy[] = {0, 0, 1, -1, 0, 0}, dz[] = {0, 0, 0, 0, 1, -1};
            double share = overflow / 6.0;
            for (int i = 0; i < 6; ++i) {
                int nx = x + dx[i], ny = y + dy[i], nz = z + dz[i];
                if (nx >= 0 && nx < size_ && ny >= 0 && ny < size_ && nz >= 0 && nz < size_) {
                    get(nx, ny, nz).mass_density += share;
                    get(nx, ny, nz).vx += dx[i] * 0.001; 
                }
            }
        } else if (cell.mass_density > PHASE_TRANSITION_THRESHOLD) {
            cell.potential_energy_leak = (cell.mass_density - PHASE_TRANSITION_THRESHOLD) * 0.01;
        }
    }

    double calculate_total_mass() const {
        double total = 0;
        for (const auto& cell : cells_) {
            total += cell.mass_density;
        }
        return total;
    }

private:
    int size_;
    std::vector<SpacetimeCell> cells_;
    std::vector<SpacetimeCell> buffer_;
};

} // namespace KG

#endif // KERNEL_CORE_HPP
