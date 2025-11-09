// Multigrid_embedded.cu
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

#define FLUID     0
#define DIRICHLET 1
#define SOLID     2

// ===================== GPU Kernels ===================== //

__global__
void smooth_redblack(double* phi, const double* rhs,
    const unsigned char* mask,
    const double* phi_bc,
    int nx, int ny, double h2, int redblack)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || j <= 0 || i >= nx - 1 || j >= ny - 1) return;
    if ((i + j) & 1 != redblack) return;

    int id = j * nx + i;

    if (mask[id] == FLUID) {
        double sum = phi[id - nx] + phi[id + nx] + phi[id - 1] + phi[id + 1];
        phi[id] = 0.25 * (sum - h2 * rhs[id]);
    }
    else if (mask[id] == DIRICHLET) {
        phi[id] = phi_bc[id];
    }
}

__global__
void residual_kernel(const double* phi, const double* rhs,
    const unsigned char* mask,
    const double* phi_bc,
    double* res,
    int nx, int ny, double h2i)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || j <= 0 || i >= nx - 1 || j >= ny - 1) return;
    int id = j * nx + i;

    if (mask[id] == FLUID) {
        double lap = phi[id - nx] + phi[id + nx] + phi[id - 1] + phi[id + 1] - 4.0 * phi[id];
        res[id] = rhs[id] - h2i * lap;
    }
    else {
        res[id] = 0.0;
    }
}

__global__
void restrict_kernel(const double* fine, double* coarse,
    const unsigned char* mask_fine,
    unsigned char* mask_coarse,
    int nxf, int nyf, int nxc, int nyc)
{
    int ic = blockIdx.x * blockDim.x + threadIdx.x;
    int jc = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic <= 0 || jc <= 0 || ic >= nxc - 1 || jc >= nyc - 1) return;

    int i = 2 * ic;
    int j = 2 * jc;
    int idc = jc * nxc + ic;
    int idf = j * nxf + i;

    // Mask coarsening: if any child is Dirichlet → Dirichlet
    unsigned char m = mask_fine[idf];
    mask_coarse[idc] = m;

    if (m == FLUID) {
        double sum = 0.0;
        sum += fine[idf];
        sum += 0.5 * (fine[idf - 1] + fine[idf + 1] + fine[idf - nxf] + fine[idf + nxf]);
        sum += 0.25 * (fine[idf - nxf - 1] + fine[idf - nxf + 1] + fine[idf + nxf - 1] + fine[idf + nxf + 1]);
        coarse[idc] = sum / 4.0;
    }
    else {
        coarse[idc] = 0.0;
    }
}

__global__
void prolongate_add_kernel(const double* coarse, double* fine,
    const unsigned char* mask_fine,
    int nxf, int nyf, int nxc, int nyc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || j <= 0 || i >= nxf - 1 || j >= nyf - 1) return;

    int ic = i / 2, jc = j / 2;
    int idc = jc * nxc + ic;
    int idf = j * nxf + i;

    if (mask_fine[idf] == FLUID) {
        fine[idf] += coarse[idc]; // simple injection
    }
}

// ===================== Host wrappers ===================== //

struct MGLevel {
    int nx, ny;
    double* phi, * rhs, * res;
    unsigned char* mask;
    double* phi_bc;
};

class Multigrid {
public:
    std::vector<MGLevel> levels;
    double h;
    int max_levels;

    Multigrid(int nx, int ny, int levels_in, double h_in) {
        h = h_in;
        max_levels = levels_in;
        int nxf = nx, nyf = ny;
        for (int l = 0; l < levels_in; l++) {
            MGLevel L;
            L.nx = nxf; L.ny = nyf;
            size_t N = nxf * nyf;
            cudaMalloc(&L.phi, N * sizeof(double));
            cudaMalloc(&L.rhs, N * sizeof(double));
            cudaMalloc(&L.res, N * sizeof(double));
            cudaMalloc(&L.mask, N * sizeof(unsigned char));
            cudaMalloc(&L.phi_bc, N * sizeof(double));
            cudaMemset(L.phi, 0, N * sizeof(double));
            cudaMemset(L.rhs, 0, N * sizeof(double));
            cudaMemset(L.res, 0, N * sizeof(double));
            cudaMemset(L.mask, 0, N * sizeof(unsigned char));
            cudaMemset(L.phi_bc, 0, N * sizeof(double));
            levels.push_back(L);
            nxf = (nxf - 1) / 2 + 1;
            nyf = (nyf - 1) / 2 + 1;
        }
    }

    ~Multigrid() {
        for (auto& L : levels) {
            cudaFree(L.phi);
            cudaFree(L.rhs);
            cudaFree(L.res);
            cudaFree(L.mask);
            cudaFree(L.phi_bc);
        }
    }

    void vcycle(int l) {
        MGLevel& L = levels[l];
        int nx = L.nx, ny = L.ny;
        dim3 threads(8, 8);
        dim3 blocks((nx + 7) / 8, (ny + 7) / 8);
        double h2 = h * h * pow(2, l) * pow(2, l);
        double h2i = 1.0 / h2;

        // pre-smooth
        for (int k = 0; k < 2; k++) {
            smooth_redblack << <blocks, threads >> > (L.phi, L.rhs, L.mask, L.phi_bc, nx, ny, h2, 0);
            smooth_redblack << <blocks, threads >> > (L.phi, L.rhs, L.mask, L.phi_bc, nx, ny, h2, 1);
        }

        if (l == max_levels - 1) return;

        // residual
        residual_kernel << <blocks, threads >> > (L.phi, L.rhs, L.mask, L.phi_bc, L.res, nx, ny, h2i);

        // restrict
        MGLevel& Lc = levels[l + 1];
        dim3 blocksC((Lc.nx + 7) / 8, (Lc.ny + 7) / 8);
        restrict_kernel << <blocksC, threads >> > (L.res, Lc.rhs, L.mask, Lc.mask, nx, ny, Lc.nx, Lc.ny);
        cudaMemset(Lc.phi, 0, Lc.nx * Lc.ny * sizeof(double));

        vcycle(l + 1);

        // prolongate
        prolongate_add_kernel << <blocks, threads >> > (Lc.phi, L.phi, L.mask, nx, ny, Lc.nx, Lc.ny);

        // post-smooth
        for (int k = 0; k < 2; k++) {
            smooth_redblack << <blocks, threads >> > (L.phi, L.rhs, L.mask, L.phi_bc, nx, ny, h2, 0);
            smooth_redblack << <blocks, threads >> > (L.phi, L.rhs, L.mask, L.phi_bc, nx, ny, h2, 1);
        }
    }
};
