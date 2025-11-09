#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernels.cuh>
#include <stdio.h>
#include <math.h>


__global__ void update_coordinates(
    double2* __restrict__ positions_new,
    double2* __restrict__ positions_old,
    const double2* __restrict__ velocities,
    const int num_particles,
    const double dt) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles) return;  // Prevent out-of-bounds access

    positions_old[idx].x = positions_new[idx].x; // Copy old position to new position
    positions_old[idx].y = positions_new[idx].y; // Copy old position to new position
    positions_new[idx].x = positions_old[idx].x + velocities[idx].x * dt;
    positions_new[idx].y = positions_old[idx].y + velocities[idx].y * dt;
}


__global__ void updateDensity(double* d_rho,
    const double2* d_positions,
    const double* d_q_type,
    const int* d_part_type,
    int m,  // number of grid points in x-direction
    int n,  // number of grid points in y-direction
    double dx,
    double dy,
    int last_alive) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= last_alive) return;

    // Get particle type and its charge
    int p_type = d_part_type[idx];
    double charge = d_q_type[p_type];
    
    // Get particle position
    const double2 pos = d_positions[idx];
    const double x = pos.x / dx;
    const double y = pos.y / dy;
    
    // Compute grid indices with bounds checking
    const int x0 = (int)floor(x);
    const int y0 = (int)floor(y);
    
    // Skip particles outside the grid
    if (x0 < 0 || x0 >= m-1 || y0 < 0 || y0 >= n-1) return;
    
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    
    // Compute interpolation weights
    const double wx = x - x0;
    const double wy = y - y0;
    const double wx_1 = 1.0 - wx;  
    const double wy_1 = 1.0 - wy;
    
    // Update density for all 4 contributing grid points
    atomicAdd(&d_rho[y0 * m + x0], charge * wx_1 * wy_1);
    atomicAdd(&d_rho[y0 * m + x1], charge * wx * wy_1);
    atomicAdd(&d_rho[y1 * m + x0], charge * wx_1 * wy);
    atomicAdd(&d_rho[y1 * m + x1], charge * wx * wy);
}

__global__ void updateParticleVelocities(double2* velocities,
    const double2* E_grid,
    const double2* positions,
    const int* part_type,
    const double* charge,
    const double* mass,
    int num_particles,
    double dx, double dy,
    int m, int n,
    double dt) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Get particle type and its charge
    int p_type = part_type[idx];
    double q = charge[p_type];
    double m_inv = 1.0 / mass[p_type];
    
    // Get particle position
    const double2 pos = positions[idx];
    const double x = pos.x / dx;
    const double y = pos.y / dy;
    
    // Compute grid indices with bounds checking
    const int x0 = (int)floor(x);
    const int y0 = (int)floor(y);
    
    // Skip particles outside the grid
    if (x0 < 0 || x0 >= m-1 || y0 < 0 || y0 >= n-1) return;
    
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    
    // Compute interpolation weights
    const double wx = x - x0;
    const double wy = y - y0;
    const double wx_1 = 1.0 - wx;  
    const double wy_1 = 1.0 - wy;

    double Ex = 0.0;
    double Ey = 0.0;

    double w0 = wx_1 * wy_1;
    double w1 = wx * wy_1;
    double w2 = wx_1 * wy;
    double w3 = wx * wy;

    Ex = w0 * E_grid[y0 * m + x0].x + 
         w1 * E_grid[y0 * m + x1].x +
         w2 * E_grid[y1 * m + x0].x +
         w3 * E_grid[y1 * m + x1].x;

    Ey = w0 * E_grid[y0 * m + x0].y +
         w1 * E_grid[y0 * m + x1].y +
         w2 * E_grid[y1 * m + x0].y +
         w3 * E_grid[y1 * m + x1].y;

    velocities[idx].x += q * Ex * m_inv * dt;
    velocities[idx].y += q * Ey * m_inv * dt;
}


__global__ void computeElectricField(double2 *E_grid, const double *phi, int m, int n, double dx, double dy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= m || y >= n) return;
    
    double Ex, Ey;
    
    // Corner cases - need to handle both x and y boundaries
    if (x == 0 && y == 0) {
        // Top-left corner
        Ex = -(phi[y * m + (x + 1)] - phi[y * m + x]) / dx;
        Ey = -(phi[(y + 1) * m + x] - phi[y * m + x]) / dy;
    }
    else if (x == m - 1 && y == 0) {
        // Top-right corner  
        Ex = -(phi[y * m + x] - phi[y * m + (x - 1)]) / dx;
        Ey = -(phi[(y + 1) * m + x] - phi[y * m + x]) / dy;
    }
    else if (x == 0 && y == n - 1) {
        // Bottom-left corner
        Ex = -(phi[y * m + (x + 1)] - phi[y * m + x]) / dx;
        Ey = -(phi[y * m + x] - phi[(y - 1) * m + x]) / dy;
    }
    else if (x == m - 1 && y == n - 1) {
        // Bottom-right corner
        Ex = -(phi[y * m + x] - phi[y * m + (x - 1)]) / dx;
        Ey = -(phi[y * m + x] - phi[(y - 1) * m + x]) / dy;
    }
    // Edge cases
    else if (x == 0) {
        // Left edge
        Ex = -(phi[y * m + (x + 1)] - phi[y * m + x]) / dx;
        Ey = -(phi[(y + 1) * m + x] - phi[(y - 1) * m + x]) / (2 * dy);
    }
    else if (x == m - 1) {
        // Right edge
        Ex = -(phi[y * m + x] - phi[y * m + (x - 1)]) / dx;
        Ey = -(phi[(y + 1) * m + x] - phi[(y - 1) * m + x]) / (2 * dy);
    }
    else if (y == 0) {
        // Top edge
        Ex = -(phi[y * m + (x + 1)] - phi[y * m + (x - 1)]) / (2 * dx);
        Ey = -(phi[(y + 1) * m + x] - phi[y * m + x]) / dy;
    }
    else if (y == n - 1) {
        // Bottom edge
        Ex = -(phi[y * m + (x + 1)] - phi[y * m + (x - 1)]) / (2 * dx);
        Ey = -(phi[y * m + x] - phi[(y - 1) * m + x]) / dy;
    }
    else {
        // Interior points - central difference
        Ex = -(phi[y * m + (x + 1)] - phi[y * m + (x - 1)]) / (2 * dx);
        Ey = -(phi[(y + 1) * m + x] - phi[(y - 1) * m + x]) / (2 * dy);
    }
    
    E_grid[y * m + x] = make_double2(Ex, Ey);
}




__global__ void compute_speeds_and_energies(
    const double2* __restrict__ V,
    double* __restrict__ V_abs,
    double* __restrict__ energies,
    const int* __restrict__ type_arr,
    const double* __restrict__ m_type,
    const int N
) {                                     
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    double vxi = V[i].x;
    double vyi = V[i].y;
    double speed_sq = vxi * vxi + vyi * vyi;
    int type = type_arr[i];
    double m = m_type[type];
    V_abs[i] = sqrt(speed_sq);
    energies[i] = 0.5 * m * speed_sq * 6.25e18; // Convert to eV

}


__global__
void interpolate_cross_sections(
    const double* __restrict__ energies,
    const double* __restrict__ xs_energy,
    const double* __restrict__ xs_values,
    double* __restrict__ output,
    const int xs_len,
    const int N
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    double E = energies[i];

    // Handle outside bounds (clip to first or last)
    if (E <= xs_energy[0]) {
        output[i] = xs_values[0];
        return;
    } else if (E >= xs_energy[xs_len - 1]) {
        output[i] = xs_values[xs_len - 1];
        return;
    }

    // Binary search for correct interval
    int low = 0;
    int high = xs_len - 2;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (E < xs_energy[mid]) {
            high = mid - 1;
                             
        } else if (E >= xs_energy[mid + 1]) {
            low = mid + 1;
        } else {
            // interpolate
            double x0 = xs_energy[mid];
            double x1 = xs_energy[mid + 1];
            double y0 = xs_values[mid];
            double y1 = xs_values[mid + 1];
            double t = (E - x0) / (x1 - x0);
            output[i] = y0 + t * (y1 - y0);
            
            return;
        }
    }
}



// CUDA kernel to convert double2 positions to float2 for OpenGL
__global__ void convertPositionsToFloat(const double2* __restrict__ d_positions, float2* __restrict__ f_positions, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        f_positions[idx].x = (float)d_positions[idx].x;
        f_positions[idx].y = (float)d_positions[idx].y;
    }
}