#pragma once
#include "particles.cuh"
#include <vector>
#include <string>

struct cross_sections{
    double* energies;  // Energies for cross sections
    double* sigmas;    // Cross sections
    double* sigmas_elastic; // Elastic cross sections
    double* sigmas_ionization; // Ionization cross sections
    double* sigmas_excitation; // Excitation cross sections
    double* sigmas_recombination; // Recombination cross sections
    double* sigmas_attachment; // Attachment cross sections
    double* sigmas_transport; // Transport cross sections
    int size;         // Size of the cross sections array
};


std::pair<std::vector<double>, std::vector<double>> read_cross_section(const std::string& filename);

void elasticCollisions(Particles2D& particles, double density, double dt, cross_sections& crossections);