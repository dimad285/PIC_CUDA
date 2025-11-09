#include "mcc.hpp"
#include "particles.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <kernels.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

using namespace std;

pair<vector<double>, vector<double>> read_cross_section(const std::string& filename) {
    vector<double> energies;
    vector<double> sigmas;

    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: cannot open cross-section file: " << filename << endl;
        return {energies, sigmas};
    }

    string line;
    while (getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        double energy, sigma;
        if (iss >> energy >> sigma) {
            energies.push_back(energy);
            sigmas.push_back(sigma);
        }
    }

    return {energies, sigmas};

}


__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, float *result, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        result[id] = curand_uniform(&state[id]);
    }
}

__global__ void collision_kernel(
    double2* __restrict__ v,                     // Velocity 
    double* __restrict__ v_abs,
    const double* __restrict__ cross_sections,  // Interpolated cross sections
    const double* __restrict__ random_nums,     // Pre-generated random numbers
    const double* __restrict__ random_angles,   // Pre-generated random angles
    const double density,                       // Gas density 
    const double dt,                            // Timestep 
    const int N                                 // Number of particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
      
    // Compute collision frequency
    double nu = density * cross_sections[idx] * v_abs[idx];
    
    // Collision probability
    double P = 1.0 - exp(-nu * dt);
    
    // Check for collision
    bool has_collision = random_nums[idx] < P;
    
    // Perform scattering if collision occurred
    if (has_collision) {
        double theta = random_angles[idx] * 6.28318530718; // 2 * PI
        v[idx].x = v_abs[idx] * cos(theta);
        v[idx].y = v_abs[idx] * sin(theta);
    }
}


__global__ void ionization_kernel(
    double2* __restrict__ v,
    double* __restrict__ v_abs,
    const double* __restrict__ cross_sections,
    const double* __restrict__ random_nums,
    const double density,
    const double dt,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // Compute ionization probability
    double P = cross_sections[idx] * v_abs[idx] * density * dt;

    // Check for ionization
    bool has_ionization = random_nums[idx] < P;

    // Perform ionization if it occurred
    if (has_ionization) {
        v[idx].x = 0.0;
        v[idx].y = 0.0;
    }
}

void elasticCollisions(Particles2D& particles, double density, double dt, cross_sections& crossections) {
    int N = particles.last_alive;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    compute_speeds_and_energies<<<numBlocks, blockSize>>>(particles.v, particles.v_abs, particles.energy, particles.type, particles.m, N);
    interpolate_cross_sections<<<numBlocks, blockSize>>>(particles.energy, crossections.energies, crossections.sigmas, particles.crs, crossections.size, N);
    
    // Use Host API for random number generation
    static curandGenerator_t gen = nullptr;
    static double* r_nums = nullptr;
    static double* r_angles = nullptr;
    static int allocated_size = 0;
    
    if (!gen) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    }
    
    if (allocated_size < N) {
        if (r_nums) {
            cudaFree(r_nums);
            cudaFree(r_angles);
        }
        cudaMalloc(&r_nums, N * sizeof(double));
        cudaMalloc(&r_angles, N * sizeof(double));
        allocated_size = N;
    }
    
    // Generate random numbers
    curandGenerateUniformDouble(gen, r_nums, N);
    curandGenerateUniformDouble(gen, r_angles, N);
    
    collision_kernel<<<numBlocks, blockSize>>>(particles.v, particles.v_abs, particles.crs, r_nums, r_angles, density, dt, N);
    

}


void ionizingCollisions(Particles2D& particles, double density, double dt, cross_sections& crossections) {
    int N = particles.last_alive;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    compute_speeds_and_energies<<<numBlocks, blockSize>>>(particles.v, particles.v_abs, particles.energy, particles.type, particles.m, N);
    interpolate_cross_sections<<<numBlocks, blockSize>>>(particles.energy, crossections.energies, crossections.sigmas, particles.crs, crossections.size, N);

    // Use Host API for random number generation
    static curandGenerator_t gen = nullptr;
    static double* r_nums = nullptr;
    static double* r_angles = nullptr;
    static int allocated_size = 0;

    if (!gen) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    }

    if (allocated_size < N) {
        if (r_nums) {
            cudaFree(r_nums);
            cudaFree(r_angles);
        }
        cudaMalloc(&r_nums, N * sizeof(double));
        cudaMalloc(&r_angles, N * sizeof(double));
        allocated_size = N;
    }

    // Generate random numbers
    curandGenerateUniformDouble(gen, r_nums, N);
    curandGenerateUniformDouble(gen, r_angles, N);

    //ionization_kernel<<<numBlocks, blockSize>>>(particles.v, particles.v_abs, particles.crs, r_nums, r_angles, density, dt, N);
}