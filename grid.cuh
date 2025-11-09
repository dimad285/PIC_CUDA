#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <Eigen/Eigen>

class Grid2D {

public:
    double* rho;
    double* phi;
    double2* E;
    int m = 0;
    int n = 0;
    double dx = 0.0;
    double dy = 0.0;
    double x = 0.0;
    double y = 0.0;

    Grid2D(int M, int N, double X, double Y);
    ~Grid2D();

    void debugGrid();
};


