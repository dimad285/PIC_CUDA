#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//#define BLOCK_SIZE_X 16
//#define BLOCK_SIZE_Y 16

//__constant__ double c_dt;
//__constant__ double c_dx;
//__constant__ double c_dy;
//__constant__ int c_m;
//__constant__ int c_n;

__global__ void update_coordinates(
    double2* __restrict__ positions_new,
    double2* __restrict__ positions_old,
    const double2* __restrict__ velocities,
    const int num_particles,
    const double dt);


__global__ void updateDensity(double* d_rho,
    const double2* d_positions,
    const double* d_q_type,
    const int* d_part_type,
    int m,
    int n,
    double dx,
    double dy,
    int last_alive);

__global__ void updateParticleVelocities(double2* velocities,
    const double2* E_grid,
    const double2* positions,
    const int* part_types,
    const double* charge,
    const double* mass,
    int num_particles,
    double dx, double dy,
    int m, int n,
    double dt);



__global__ void computeElectricField(double2* E_grid,
    const double* phi,
    int m,
    int n,
    double dx,
    double dy);


__global__ void compute_speeds_and_energies(
    const double2* __restrict__ V,
    double* __restrict__ V_abs,
    double* __restrict__ energies,
    const int* __restrict__ type_arr,
    const double* __restrict__ m_type,
    const int N
);


__global__
void interpolate_cross_sections(
    const double* __restrict__ energies,
    const double* __restrict__ xs_energy,
    const double* __restrict__ xs_values,
    double* __restrict__ output,
    const int xs_len,
    const int N
);

__global__ 
void convertPositionsToFloat(
    const double2* __restrict__ d_positions, 
    float2* __restrict__ f_positions,
    int count);