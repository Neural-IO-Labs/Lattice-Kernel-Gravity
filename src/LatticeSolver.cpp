#include "../include/KernelCore.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

namespace KG {

class LatticeSolver {
public:
    LatticeSolver(Lattice& lattice) : lattice_(lattice) {
        monitor_log_.open("echoes.csv");
        monitor_log_ << "step,signal\n";
    }

    ~LatticeSolver() {
        if (monitor_log_.is_open()) monitor_log_.close();
    }

    void step(double dt) {
        int size = lattice_.size();

        // --- Pass 1: Advection (Moving mass and momentum) ---
        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y) {
                for (int z = 0; z < size; ++z) {
                    lattice_.set_buffer(x, y, z, lattice_.get_read(x, y, z));
                }
            }
        }

        for (int x = 1; x < size - 1; ++x) {
            for (int y = 1; y < size - 1; ++y) {
                for (int z = 1; z < size - 1; ++z) {
                    const SpacetimeCell& current = lattice_.get_read(x, y, z);
                    double dt_local = dt * current.clock_speed;
                    double out_ratio = (std::abs(current.vx) * dt_local);
                    if (out_ratio > 0.4) out_ratio = 0.4;

                    // Outgoing density
                    SpacetimeCell target = current;
                    target.mass_density -= current.mass_density * out_ratio;
                    lattice_.set_buffer(x, y, z, target);

                    // Incoming density (Push to neighbors in buffer)
                    if (current.vx > 0) {
                        double flux = current.mass_density * out_ratio;
                        SpacetimeCell& neighbor = lattice_.get_buffer_ref(x + 1, y, z);
                        neighbor.mass_density += flux;
                        neighbor.vx = (neighbor.vx * neighbor.mass_density + current.vx * flux) / (neighbor.mass_density + 1e-25);
                    } else if (current.vx < 0) {
                        double flux = current.mass_density * out_ratio;
                        SpacetimeCell& neighbor = lattice_.get_buffer_ref(x - 1, y, z);
                        neighbor.mass_density += flux;
                        neighbor.vx = (neighbor.vx * neighbor.mass_density + current.vx * flux) / (neighbor.mass_density + 1e-25);
                    }
                }
            }
        }
        lattice_.swap_buffers();

        // --- Pass 2: Density Gate (Strictly Conservative) ---
        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y) {
                for (int z = 0; z < size; ++z) {
                    lattice_.set_buffer(x, y, z, lattice_.get_read(x, y, z));
                }
            }
        }

        for (int x = 1; x < size - 1; ++x) {
            for (int y = 1; y < size - 1; ++y) {
                for (int z = 1; z < size - 1; ++z) {
                    const SpacetimeCell& current = lattice_.get_read(x, y, z);
                    if (current.mass_density > D_MAX) {
                        double overflow = current.mass_density - D_MAX;
                        double share = overflow / 6.0;
                        
                        // Fix the source in buffer
                        SpacetimeCell& source = lattice_.get_buffer_ref(x, y, z);
                        source.mass_density = D_MAX;
                        source.clock_speed = 1.0 / (1.0 + (overflow / D_MAX) * 1e12);

                        // Distribute to neighbors in buffer
                        int dx[] = {1, -1, 0, 0, 0, 0}, dy[] = {0, 0, 1, -1, 0, 0}, dz[] = {0, 0, 0, 0, 1, -1};
                        for (int i = 0; i < 6; ++i) {
                            SpacetimeCell& neighbor = lattice_.get_buffer_ref(x+dx[i], y+dy[i], z+dz[i]);
                            neighbor.mass_density += share;
                            neighbor.vx += dx[i] * 0.001; 
                        }
                    }
                }
            }
        }
        lattice_.swap_buffers();

        static int step_count = 0;
        record_boundary_flux(step_count++);
    }

private:
    Lattice& lattice_;
    std::ofstream monitor_log_;

    void record_boundary_flux(int step) {
        int size = lattice_.size();
        int cx = size / 2, cy = size / 2, cz = size / 2;
        int R = size / 3;
        double total_flux = 0;
        for (int dx = -R; dx <= R; ++dx) {
            for (int dy = -R; dy <= R; ++dy) {
                for (int dz = -R; dz <= R; ++dz) {
                    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (std::abs(dist - (double)R) < 1.0) {
                        total_flux += lattice_.get_read(cx+dx, cy+dy, cz+dz).mass_density;
                    }
                }
            }
        }
        monitor_log_ << step << "," << total_flux << "\n";
    }
};

} // namespace KG
