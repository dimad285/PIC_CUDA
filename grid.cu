#include "grid.cuh"
#include <cuda_runtime.h>
#include <kernels.cuh> 
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/Eigen>
#include <iostream>
#include <cublas_v2.h>

Grid2D::Grid2D(int M, int N, double X, double Y) {
    m = M;
    n = N;
    x = X;
    y = Y;
    dx = x / (m - 1);
    dy = y / (n - 1);
    cudaMalloc(&rho, m * n * sizeof(double));
    cudaMalloc(&phi, m * n * sizeof(double));
    cudaMalloc(&E, m * n * sizeof(double2));
}

Grid2D::~Grid2D(){
    cudaFree(rho);
    cudaFree(phi);
    cudaFree(E);

}


void Grid2D::debugGrid() {
    std::vector<double> h_rho(m * n);
    cudaMemcpy(h_rho.data(), rho, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Grid density (rho):" << std::endl;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            std::cout << h_rho[j * m + i] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<double> h_phi(m * n);
    cudaMemcpy(h_phi.data(), phi, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << std::endl;
    std::cout << "Grid potential (phi):" << std::endl;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            std::cout << h_phi[j * m + i] << " ";
        }
        std::cout << std::endl;
    }
}

