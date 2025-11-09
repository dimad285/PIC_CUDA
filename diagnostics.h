#pragma once
#include <iostream>
#include <iomanip>

struct SimulationTiming {
    double coordinate_update_us = 0.0;
    double collision_detection_us = 0.0; // Placeholder for collision detection timing
    double particle_removal = 0.0; // Placeholder for particle removal timing
    double density_reset_us = 0.0;
    double density_update_us = 0.0;
    double solver_us = 0.0;
    double electric_field_us = 0.0;
    double velocity_update_us = 0.0;
    double synchronize_us = 0.0;
    double mcc_us = 0.0; // Placeholder for Monte Carlo collision timing
    double total_us = 0.0;
};


inline void printDetailedTiming(const SimulationTiming& timing, int steps, int particles) {
    std::cout << "\n=== DETAILED SIMULATION TIMING (microseconds) ===" << std::endl;
    std::cout << "Particles: " << particles << ", Steps: " << steps << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    // Per-step timings
    std::cout << "\nPer-step average timings:" << std::endl;
    std::cout << "  Coordinate Update:  " << std::setw(8) << timing.coordinate_update_us / steps << " us" << std::endl;
    std::cout << "  Collision Detection: " << std::setw(8) << timing.collision_detection_us / steps << " us" << std::endl;
    std::cout << "  Particle Removal:   " << std::setw(8) << timing.particle_removal / steps << " us" << std::endl;
    std::cout << "  Density Reset:      " << std::setw(8) << timing.density_reset_us / steps << " us" << std::endl;
    std::cout << "  Density Update:     " << std::setw(8) << timing.density_update_us / steps << " us" << std::endl;

    if (timing.solver_us > 0) {
        std::cout << "  Solver:             " << std::setw(8) << timing.solver_us / steps << " us" << std::endl;
    }
    if (timing.electric_field_us > 0) {
        std::cout << "  Electric Field:     " << std::setw(8) << timing.electric_field_us / steps << " us" << std::endl;
    }
    if (timing.velocity_update_us > 0) {
        std::cout << "  Velocity Update:    " << std::setw(8) << timing.velocity_update_us / steps << " us" << std::endl;
    }
    if (timing.mcc_us > 0) {
        std::cout << "  Monte Carlo Collisions: " << std::setw(8) << timing.mcc_us / steps << " us" << std::endl;
    }

    std::cout << "  Final Sync:         " << std::setw(8) << timing.synchronize_us << " us" << std::endl;
    std::cout << "  TOTAL:              " << std::setw(8) << timing.total_us << " us" << std::endl;

    // Total timings for all steps
    std::cout << "\nTotal timings (all " << steps << " steps):" << std::endl;
    std::cout << "  Coordinate Update:  " << std::setw(8) << timing.coordinate_update_us << " us" << std::endl;
    std::cout << "  Density Operations: " << std::setw(8) << timing.density_reset_us + timing.density_update_us << " us" << std::endl;

    // Performance metrics
    double particles_per_microsecond = (double)(particles * steps) / timing.total_us;
    std::cout << "\nPerformance metrics:" << std::endl;
    std::cout << "  Particles/us:       " << std::setw(8) << particles_per_microsecond << std::endl;
    std::cout << "  Million particles/s:" << std::setw(8) << particles_per_microsecond << std::endl;

    std::cout << "================================================\n" << std::endl;
}