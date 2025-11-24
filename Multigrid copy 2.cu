// Multigrid.cu - Version with Persistent Memory Allocation
// This version eliminates repeated malloc/free overhead by caching allocations

#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>  
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include "Multigrid.hpp"
#include <stdexcept>
#include <mutex>

using namespace std;
using namespace cooperative_groups;

// ========== PERSISTENT MEMORY MANAGEMENT ==========

struct MultigridLevel {
    double* phi;
    double* rhs;
    double* res;
    int nx;
    int ny;
    size_t size_bytes;
};

class MultigridMemoryPool {
private:
    vector<MultigridLevel> levels;
    double* res_norm_sq;
    double* rhs_norm_sq;
    bool* d_converged;
    double* relative_residual_out;
    
    int allocated_levels;
    int allocated_nx;
    int allocated_ny;
    bool is_initialized;
    
    std::mutex allocation_mutex;  // Thread safety
    
public:
    MultigridMemoryPool() : 
        res_norm_sq(nullptr),
        rhs_norm_sq(nullptr),
        d_converged(nullptr),
        relative_residual_out(nullptr),
        allocated_levels(0),
        allocated_nx(0),
        allocated_ny(0),
        is_initialized(false) {}
    
    ~MultigridMemoryPool() {
        cleanup();
    }
    
    // Check if we need to reallocate
    bool needs_reallocation(int nx, int ny, int num_levels) {
        return !is_initialized || 
               allocated_nx != nx || 
               allocated_ny != ny || 
               allocated_levels != num_levels;
    }
    
    // Allocate all memory needed for multigrid
    void allocate(int nx, int ny, int num_levels) {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        
        // Check if we can reuse existing allocation
        if (!needs_reallocation(nx, ny, num_levels)) {
            return;  // Memory already allocated with correct sizes
        }
        
        // Clean up old allocation if it exists
        if (is_initialized) {
            cleanup_internal();
        }
        
        levels.resize(num_levels);
        
        int n = nx;
        for (int l = 0; l < num_levels; l++) {
            levels[l].nx = n;
            levels[l].ny = n;
            levels[l].size_bytes = n * n * sizeof(double);
            
            // Allocate memory for this level
            cudaError_t err;
            err = cudaMalloc(&levels[l].phi, levels[l].size_bytes);
            if (err != cudaSuccess) {
                cerr << "Failed to allocate phi at level " << l << ": " 
                     << cudaGetErrorString(err) << endl;
                cleanup_internal();
                throw runtime_error("CUDA allocation failed");
            }
            
            err = cudaMalloc(&levels[l].rhs, levels[l].size_bytes);
            if (err != cudaSuccess) {
                cerr << "Failed to allocate rhs at level " << l << ": " 
                     << cudaGetErrorString(err) << endl;
                cleanup_internal();
                throw runtime_error("CUDA allocation failed");
            }
            
            err = cudaMalloc(&levels[l].res, levels[l].size_bytes);
            if (err != cudaSuccess) {
                cerr << "Failed to allocate res at level " << l << ": " 
                     << cudaGetErrorString(err) << endl;
                cleanup_internal();
                throw runtime_error("CUDA allocation failed");
            }
            
            // Initialize to zero
            cudaMemset(levels[l].phi, 0, levels[l].size_bytes);
            cudaMemset(levels[l].rhs, 0, levels[l].size_bytes);
            cudaMemset(levels[l].res, 0, levels[l].size_bytes);
            
            n = (n + 1) / 2;  // Next coarser level
        }
        
        // Allocate auxiliary arrays for norms and convergence
        cudaMalloc(&res_norm_sq, sizeof(double));
        cudaMalloc(&rhs_norm_sq, sizeof(double));
        cudaMalloc(&d_converged, sizeof(bool));
        cudaMalloc(&relative_residual_out, sizeof(double));
        
        allocated_levels = num_levels;
        allocated_nx = nx;
        allocated_ny = ny;
        is_initialized = true;
    }
    
    // Get pointers to level data
    MultigridLevel& get_level(int l) {
        if (l < 0 || l >= allocated_levels) {
            throw runtime_error("Invalid multigrid level requested");
        }
        return levels[l];
    }
    
    // Get auxiliary pointers
    double* get_res_norm_sq() { return res_norm_sq; }
    double* get_rhs_norm_sq() { return rhs_norm_sq; }
    bool* get_converged_flag() { return d_converged; }
    double* get_relative_residual() { return relative_residual_out; }
    
    int get_num_levels() const { return allocated_levels; }
    
    // Cleanup all allocated memory
    void cleanup() {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        cleanup_internal();
    }
    
private:
    void cleanup_internal() {
        if (!is_initialized) return;
        
        for (int l = 0; l < allocated_levels; l++) {
            if (levels[l].phi) cudaFree(levels[l].phi);
            if (levels[l].rhs) cudaFree(levels[l].rhs);
            if (levels[l].res) cudaFree(levels[l].res);
        }
        
        if (res_norm_sq) cudaFree(res_norm_sq);
        if (rhs_norm_sq) cudaFree(rhs_norm_sq);
        if (d_converged) cudaFree(d_converged);
        if (relative_residual_out) cudaFree(relative_residual_out);
        
        levels.clear();
        res_norm_sq = nullptr;
        rhs_norm_sq = nullptr;
        d_converged = nullptr;
        relative_residual_out = nullptr;
        
        allocated_levels = 0;
        allocated_nx = 0;
        allocated_ny = 0;
        is_initialized = false;
    }
};

// Global persistent memory pool (one per process)
static MultigridMemoryPool g_multigrid_pool;

// ========== ORIGINAL CUDA KERNELS (UNCHANGED) ==========

__global__
void gauss_seidel_red_black(
    double* phi,
    double* rho,
    double h2,
    double omega,
    int nx,
    int ny,
    int is_red
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny;
    if (idx >= size) return;
    int i = idx % nx;
    int j = idx / nx;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        if ((i + j) % 2 == is_red) {
            int idx_n = (j + 1) * nx + i;
            int idx_s = (j - 1) * nx + i;
            int idx_e = j * nx + (i + 1);
            int idx_w = j * nx + (i - 1);

            double update = 0.25 * (
                phi[idx_n] + phi[idx_s] +
                phi[idx_e] + phi[idx_w] -
                h2 * rho[idx]
            );
            phi[idx] = (1.0 - omega) * phi[idx] + omega * update;
        }
    }
}

__global__ void gauss_seidel_red_black_sync(
    double* phi,
    double* rho,
    double h2,
    double omega,
    int nx,
    int ny,
    int num_iterations
) {
    grid_group grid = this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny;
    
    int i = idx % nx;
    int j = idx / nx;
    
    for (int iter = 0; iter < num_iterations; iter++) {
        // Red points
        if (idx < size && i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            if ((i + j) % 2 == 1) {
                int idx_n = (j + 1) * nx + i;
                int idx_s = (j - 1) * nx + i;
                int idx_e = j * nx + (i + 1);
                int idx_w = j * nx + (i - 1);
                double update = 0.25 * (
                    phi[idx_n] + phi[idx_s] +
                    phi[idx_e] + phi[idx_w] -
                    h2 * rho[idx]
                );
                phi[idx] = (1.0 - omega) * phi[idx] + omega * update;
            }
        }

        grid.sync();
        
        // Black points
        if (idx < size && i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            if ((i + j) % 2 == 0) {
                int idx_n = (j + 1) * nx + i;
                int idx_s = (j - 1) * nx + i;
                int idx_e = j * nx + (i + 1);
                int idx_w = j * nx + (i - 1);
                double update = 0.25 * (
                    phi[idx_n] + phi[idx_s] +
                    phi[idx_e] + phi[idx_w] -
                    h2 * rho[idx]
                );
                phi[idx] = (1.0 - omega) * phi[idx] + omega * update;
            }
        }
        
        grid.sync();
    }
}

__global__
void residual_kernel(
    double* phi,
    double* rho,
    double* res,
    double inv_h2,
    int nx,
    int ny
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;

    int i = idx % nx;
    int j = idx / nx;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int index = j * nx + i;
        double laplacian = (
            phi[index - nx] + phi[index + nx] +
            phi[index - 1] + phi[index + 1] -
            4.0 * phi[index]
        ) * inv_h2;
        res[index] = rho[index] - laplacian;
    } else {
        res[idx] = 0.0;
    }
}

__global__
void restrict_kernel(const double* fine, double* coarse, int nx_fine, int ny_fine, int nx_coarse, int ny_coarse) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny_coarse || j >= nx_coarse) return;

    int i_f = 2 * i;
    int j_f = 2 * j;

    double result = 0.0;
    double weights[3][3] = {{0.0625, 0.125, 0.0625},
                             {0.125,  0.25,  0.125},
                             {0.0625, 0.125, 0.0625}};

    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            int ii = i_f + di;
            int jj = j_f + dj;
            if (ii >= 0 && ii < ny_fine && jj >= 0 && jj < nx_fine) {
                result += fine[ii * nx_fine + jj] * weights[di + 1][dj + 1];
            }
        }
    }

    coarse[i * nx_coarse + j] = result;
}

__global__
void prolong_kernel(double* fine, const double* coarse, int nx_coarse, int ny_coarse, int nx_fine) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / nx_fine;
    int j = idx % nx_fine;
    if (i >= 2 * ny_coarse - 1 || j >= 2 * nx_coarse - 1) return;

    int i_c = i / 2;
    int j_c = j / 2;
    int c_idx = i_c * nx_coarse + j_c;

    double val = 0.0;

    bool is_i_odd = i % 2;
    bool is_j_odd = j % 2;

    if (!is_i_odd && !is_j_odd) {
        val = coarse[c_idx];
    } else if (is_i_odd && !is_j_odd && i_c + 1 < ny_coarse) {
        val = 0.5 * (coarse[c_idx] + coarse[(i_c + 1) * nx_coarse + j_c]);
    } else if (!is_i_odd && is_j_odd && j_c + 1 < nx_coarse) {
        val = 0.5 * (coarse[c_idx] + coarse[i_c * nx_coarse + (j_c + 1)]);
    } else if (is_i_odd && is_j_odd && i_c + 1 < ny_coarse && j_c + 1 < nx_coarse) {
        val = 0.25 * (
            coarse[c_idx] +
            coarse[(i_c + 1) * nx_coarse + j_c] +
            coarse[i_c * nx_coarse + (j_c + 1)] +
            coarse[(i_c + 1) * nx_coarse + (j_c + 1)]
        );
    }

    fine[i * nx_fine + j] += val;
}

__global__ void norm_kernel(
    double* vec,
    double* result,
    int size
) {
    extern __shared__ double shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared_data[tid] = (idx < size) ? vec[idx] * vec[idx] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

__global__ void check_convergence_kernel(
    double* res_norm_sq,
    double* rhs_norm, 
    double tolerance, 
    int size,
    bool* converged,
    double* relative_residual_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        double res_norm = sqrt(res_norm_sq[0]);
        double relative_residual = res_norm / rhs_norm[0];
        
        *relative_residual_out = relative_residual;
        *converged = (relative_residual < tolerance);
    }
}

// ========== HOST FUNCTIONS (UNCHANGED) ==========

void smooth(double* phi, double* rho, int nx, int ny, double h, double omega, int iterations) {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int block = 256;
    int grid = (nx * ny + block - 1) / block;
    double h2 = h * h;
    void* kernel_args[] = {&phi, &rho, &h2, &omega, &nx, &ny, &iterations};

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)gauss_seidel_red_black_sync,
        grid,
        block,
        kernel_args
    );

    if (err != cudaSuccess) {
        cerr << "Cooperative kernel launch failed: "
             << cudaGetErrorString(err) << endl;
    }
}

void residual(double* res, double* phi, double* rho, int nx, int ny, double h) {
    int block = 256;
    int grid = (nx * ny + block - 1) / block;
    double inv_h2 = 1.0 / (h * h);
    residual_kernel<<<grid, block>>>(phi, rho, res, inv_h2, nx, ny);
}

void restrict_level(double* coarse, double* fine, int nx_fine, int ny_fine) {
    int nx_coarse = (nx_fine + 1) / 2;
    int ny_coarse = (ny_fine + 1) / 2;
    dim3 block(16, 16);
    dim3 grid((nx_coarse + 15) / 16, (ny_coarse + 15) / 16);
    restrict_kernel<<<grid, block>>>(fine, coarse, nx_fine, ny_fine, nx_coarse, ny_coarse);
}

void prolong_level(double* fine, double* coarse, int nx_coarse, int ny_coarse) {
    int nx_fine = 2 * (nx_coarse - 1) + 1;
    int ny_fine = 2 * (ny_coarse - 1) + 1;
    int block = 256;
    int grid = (nx_fine * ny_fine + block - 1) / block;
    prolong_kernel<<<grid, block>>>(fine, coarse, nx_coarse, ny_coarse, nx_fine);
}

void l2_norm_device(double* vec_d, double* result_d, int size) {
    cudaMemset(result_d, 0, sizeof(double));
    int block = 256;
    int grid = (size + block - 1) / block;
    norm_kernel<<<grid, block, block * sizeof(double)>>>(vec_d, result_d, size);
}

void v_cycle(
    vector<double*>& phi,
    vector<double*>& rhs,
    vector<double*>& res,
    int levels,
    int nx,
    int ny,
    double h,
    double omega
) {
    int n = nx;
    double hh = h;

    for (int l = 0; l < levels - 1; ++l) {
        smooth(phi[l], rhs[l], n, n, hh, omega, 3);
        residual(res[l], phi[l], rhs[l], n, n, hh);
        restrict_level(rhs[l + 1], res[l], n, n);
        cudaMemset(phi[l + 1], 0, sizeof(double) * ((n + 1) / 2) * ((n + 1) / 2));
        n = (n + 1) / 2;
        hh *= 2;
    }
    
    smooth(phi[levels - 1], rhs[levels - 1], n, n, hh, omega, 20);

    for (int l = levels - 2; l >= 0; --l) {
        n = n * 2 - 1;
        hh /= 2;
        prolong_level(phi[l], phi[l + 1], (n + 1) / 2, (n + 1) / 2);
        smooth(phi[l], rhs[l], n, n, hh, omega, 3);
    }
}

// ========== MODIFIED SOLVER WITH PERSISTENT MEMORY ==========

void solve_multigrid(double* phi_d, double* rho_d, int nx, int ny,
    int levels, double h, double omega) {
    
    // Ensure memory pool is allocated (only allocates on first call or size change)
    g_multigrid_pool.allocate(nx, ny, levels);
    
    // Setup vectors pointing to pool memory
    vector<double*> phi(levels), rhs(levels), res(levels);
    for (int l = 0; l < levels; l++) {
        MultigridLevel& level = g_multigrid_pool.get_level(l);
        phi[l] = level.phi;
        rhs[l] = level.rhs;
        res[l] = level.res;
    }
    
    // First level uses user-provided arrays
    phi[0] = phi_d;
    rhs[0] = rho_d;
    
    // Get auxiliary pointers from pool
    double* res_norm = g_multigrid_pool.get_res_norm_sq();
    double* rhs_norm = g_multigrid_pool.get_rhs_norm_sq();
    bool* d_converged = g_multigrid_pool.get_converged_flag();
    double* relative_residual_out = g_multigrid_pool.get_relative_residual();
    
    // Prepare residual array for level 0
    double* res0 = g_multigrid_pool.get_level(0).res;
    
    // Compute RHS norm once
    l2_norm_device(rho_d, rhs_norm, nx * ny);

    // Pinned memory for async convergence check
    bool* h_converged_pinned = nullptr;
    cudaHostAlloc((void**)&h_converged_pinned, sizeof(bool), cudaHostAllocDefault);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const double tol = 1e-5;
    const int max_cycles = 50;

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        v_cycle(phi, rhs, res, levels, nx, ny, h, omega);

        // Check convergence
        cudaMemset(d_converged, 0, sizeof(bool));
        residual(res0, phi[0], rhs[0], nx, ny, h);

        l2_norm_device(res0, res_norm, nx * ny);
        check_convergence_kernel<<<1, 1>>>(res_norm, rhs_norm, tol, nx * ny, 
                                           d_converged, relative_residual_out);

        // Async copy for better overlap
        cudaMemcpyAsync(h_converged_pinned, d_converged, sizeof(bool), 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (*h_converged_pinned)
            break;
    }

    // Copy final result back to user array
    cudaMemcpy(phi_d, phi[0], nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);

    // Cleanup temporary resources
    cudaFreeHost(h_converged_pinned);
    cudaStreamDestroy(stream);
    
    // Note: We do NOT free the pool memory - it persists for next call
}