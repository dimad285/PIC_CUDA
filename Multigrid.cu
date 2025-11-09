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

using namespace std;
using namespace cooperative_groups;
// ========== CUDA KERNELS ==========

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
    
    // Calculate grid coordinates
    int i = idx % nx;
    int j = idx / nx;
    
    // CRITICAL: All threads must participate in grid.sync()
    // Don't return early - use conditionals instead
    
    for (int iter = 0; iter < num_iterations; iter++) {
        // Red points (is_red = 1)
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

        

        // ALL threads must reach this point - no early returns above!
        grid.sync();
        // Black points (is_red = 0)
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
        
        // ALL threads must reach this point too
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

__global__ void devide_kernel(
    double* devisible,
    double* denom
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 && denom[0] != 0.0) {
        devisible[0] /= denom[0];
    }
}


// Device-side convergence check kernel
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
        // Sum contributions from all blocks (simplified - you'd need atomics for multiple blocks)
        double res_norm = sqrt(res_norm_sq[0]);
        double relative_residual = res_norm / rhs_norm[0];
        
        *relative_residual_out = relative_residual;
        *converged = (relative_residual < tolerance);
    }
}
// ========== MULTIGRID SOLVER ==========

void smooth(double* phi, double* rho, int nx, int ny, double h, double omega, int iterations) {

    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    dim3 blockSize(256, 1, 1);
    dim3 gridSize(prop.multiProcessorCount, 1, 1);  // often safe starting point

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
    std::cerr << "Cooperative kernel launch failed: "
                << cudaGetErrorString(err) << std::endl;
    }
}


void _smooth(double* phi, double* rho, int nx, int ny, double h, double omega, int iterations) {
    
    int block = 256;
    int grid = (nx * ny + block - 1) / block;
    double h2 = h * h;

    for (int i = 0; i < iterations; ++i) {
        // Red pass
        gauss_seidel_red_black<<<grid, block>>>(phi, rho, h2, omega, nx, ny, 1);
        //cudaDeviceSynchronize();  // Ensure red pass completes before black pass
        
        // Black pass
        gauss_seidel_red_black<<<grid, block>>>(phi, rho, h2, omega, nx, ny, 0);
        //cudaDeviceSynchronize();  // Ensure black pass completes before next iteration
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

double l2_norm_cublas(double* vec_d, int size, cublasHandle_t handle) {
    double result = 0.0;
    cublasDnrm2(handle, size, vec_d, 1, &result);
    return result;
}

double l2_norm(double* vec_d, int size) {
    double* result_d;
    
    // Allocate device memory with error checking
    cudaError_t err = cudaMalloc(&result_d, sizeof(double));
    if (err != cudaSuccess) {
        cerr << "Error allocating device memory: " << cudaGetErrorString(err) << endl;
        return -1.0;
    }
    
    // Initialize result to zero
    cudaMemset(result_d, 0, sizeof(double));
    
    // Launch kernel
    int block = 256;
    int grid = (size + block - 1) / block;
    norm_kernel<<<grid, block, block * sizeof(double)>>>(vec_d, result_d, size);
    
    // CRITICAL: Wait for kernel to complete before proceeding
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "Kernel execution error: " << cudaGetErrorString(err) << endl;
        cudaFree(result_d);
        return -1.0;
    }
    
    // Check for any kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Kernel launch error: " << cudaGetErrorString(err) << endl;
        cudaFree(result_d);
        return -1.0;
    }
    
    // Copy result back to host
    double result;
    err = cudaMemcpy(&result, result_d, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "Error copying result to host: " << cudaGetErrorString(err) << endl;
        cudaFree(result_d);
        return -1.0;
    }
    
    // Clean up
    cudaFree(result_d);
    
    return sqrt(result);
}


void l2_norm_device(double* vec_d, double* result_d, int size) {
    /* Returns |r|^2 */
    cudaMemset(result_d, 0, sizeof(double));

    int block = 256;
    int grid = (size + block - 1) / block;
    norm_kernel<<<grid, block, block * sizeof(double)>>>(vec_d, result_d, size);
}

void check_convergence(double* res_norm_sq, double* rhs_norm, double tolerance, int size, bool* converged, double* relative_residual_out) {

    check_convergence_kernel<<<1, 1>>>(res_norm_sq, rhs_norm, tolerance, size, converged, relative_residual_out);
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


void solve_multigrid(double* phi_d, double* rho_d, int nx, int ny,
    int levels, double h, double omega) {
    int n = nx;
    size_t bytes = n * n * sizeof(double);
    vector<double*> phi(levels), rhs(levels), res(levels);

    for (int l = 0; l < levels; ++l) {
        cudaMalloc(&phi[l], bytes);
        cudaMalloc(&rhs[l], bytes);
        cudaMalloc(&res[l], bytes);
        cudaMemset(phi[l], 0, bytes);
        cudaMemset(rhs[l], 0, bytes);
        cudaMemset(res[l], 0, bytes);
        bytes = ((n + 1) / 2) * ((n + 1) / 2) * sizeof(double);
        n = (n + 1) / 2;
    }

    phi[0] = phi_d;
    rhs[0] = rho_d;
    double* res0;
    double* rhs_norm;
    double* res_norm;
    bool* d_converged;
    //bool h_converged = false;
    double* relative_residual_out;
    cudaMalloc(&relative_residual_out, sizeof(double));
    cudaMalloc(&res0, nx * ny * sizeof(double));
    cudaMalloc(&rhs_norm, sizeof(double));
    cudaMalloc(&res_norm, sizeof(double));
    cudaMalloc(&d_converged, sizeof(bool));

    bool* h_converged_pinned = nullptr;
    cudaHostAlloc((void**)&h_converged_pinned, sizeof(bool), cudaHostAllocDefault);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //residual(res0, phi[0], rhs[0], nx, ny, h);
    //double norm0 = l2_norm(rho_d, nx * ny);
    l2_norm_device(rho_d, rhs_norm, nx * ny); // |r|^2

    const double tol = 1e-5;
    const int max_cycles = 50;

    for (int cycle = 0; cycle < max_cycles; ++cycle) {

        v_cycle(phi, rhs, res, levels, nx, ny, h, omega);

        cudaMemset(d_converged, 0, sizeof(bool));
        residual(res0, phi[0], rhs[0], nx, ny, h);

        l2_norm_device(res0, res_norm, nx * ny);
        //check_convergence(res_norm, rhs_norm, tol, nx * ny, d_converged, relative_residual_out);
        check_convergence_kernel << <1, 1 >> > (res_norm, rhs_norm, tol, nx * ny, d_converged, relative_residual_out);

        //cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(h_converged_pinned, d_converged, sizeof(bool), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);  // Ensure data is ready before accessing it

        if (*h_converged_pinned)
            break;
    }
    //cout << "Converged in " << cycle + 1 << endl;
    cudaFree(res0);
    cudaMemcpy(phi_d, phi[0], nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);

    for (int l = 1; l < levels; ++l) {  // Don't free phi[0] and rhs[0] as they are user pointers
        cudaFree(phi[l]);
        cudaFree(rhs[l]);
        cudaFree(res[l]);
    }
    cudaFree(res[0]);  // But we can free res[0]
    cudaFree(rhs_norm);
    cudaFree(res_norm);
    cudaFree(d_converged);
    cudaFree(relative_residual_out);
    cudaFreeHost(h_converged_pinned);
    cudaStreamDestroy(stream);
}
