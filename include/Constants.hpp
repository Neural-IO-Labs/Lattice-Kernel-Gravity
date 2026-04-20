#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cmath>

namespace KG {

// --- Fundamental Physical Constants ---
constexpr double L_PLANCK = 1.616255e-35; // Fundamental Planck length (m)
constexpr double D_PLANCK = 5.155e96;     // Planck density limit (kg/m^3)
constexpr double G_CONST = 6.67430e-11;   // Newtonian Gravitational Constant
constexpr double H_BAR = 1.0545718e-34;   // Reduced Planck constant (Action)

// --- Spacetime Governing Thresholds ---
// D_MAX: The strict mass-density upper bound where spacetime transitions to a non-singular Quantum Solid.
constexpr double D_MAX = D_PLANCK; 
constexpr double PHASE_TRANSITION_THRESHOLD = 0.9 * D_MAX;

// --- Lattice Scaling & Resolution ---
// The simulation operates on a discretized manifold where each sector represents N Planck lengths.
// PROTON_SCALE (2.6e18): Resolves subatomic structural resonances.
// MACRO_SCALE (1e30): Resolves black hole merger gravitational echoes.
constexpr double SECTOR_SIZE_MULTIPLIER = 2.6e18; 
constexpr double L_SECTOR = L_PLANCK * SECTOR_SIZE_MULTIPLIER;

// --- Baryonic Mass Profiles ---
constexpr double M_PROTON_SCHWARZSCHILD = 8.85e11; // Effective Schwarzschild mass of a proton (kg)

} // namespace KG

#endif // CONSTANTS_HPP
