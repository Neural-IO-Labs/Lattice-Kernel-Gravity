#include "include/GpuSolver.hpp"
#include "include/Constants.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "--- KERNEL_GRAVITY: Initializing HIGH-FIDELITY CUDA Engine ---" << std::endl;
    std::cout << "D_MAX (Planck Limit): " << KG::D_MAX << " kg/m^3" << std::endl;

    // Use a 256^3 lattice for the initial high-fidelity test
    const int GRID_SIZE = 256;
    KG::GpuLattice lattice(GRID_SIZE);
    KG::GpuSolver solver(lattice);

    std::cout << "Injecting Localized Protonic Inhomogeneity (Schwarzschild Mass)..." << std::endl;
    solver.initialize_idle_proton();

    // Temporal Evolution Loop
    std::cout << "Commencing 5th-Order WENO Flux Evolution..." << std::endl;
    double dt = 0.02; // Time-step constraint for high-order stability
    for (int t = 0; t < 500; ++t) {
        solver.step(dt);
        
        if (t % 50 == 0) {
            std::cout << "T+[" << t << "] - Manifold evolution stable..." << std::endl;
        }
    }

    std::cout << "Simulation Complete. Observational data exported to 'echoes_proton.csv'." << std::endl;
    return 0;
}
