#pragma once

void solve_multigrid(
    double* phi_d,
    double* rho_d,
    int nx,
    int ny,
    int levels,
    double h,
    double omega = 1.25
);

