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


struct MultigridTiming {
    double smooth_pre_us = 0.0;
    double smooth_post_us = 0.0;
    double smooth_coarse_us = 0.0;
    double residual_us = 0.0;
    double restrict_us = 0.0;
    double prolong_us = 0.0;
    double norm_us = 0.0;
    double convergence_check_us = 0.0;
    double memset_us = 0.0;
    int num_vcycles = 0;
    
    void print_summary() {
        std::cout << "\n=== MULTIGRID DETAILED TIMING ===" << std::endl;
        std::cout << "Number of V-cycles: " << num_vcycles << std::endl;
        std::cout << "Per V-cycle breakdown:" << std::endl;
        std::cout << "  Pre-smoothing:     " << smooth_pre_us / num_vcycles << " us" << std::endl;
        std::cout << "  Post-smoothing:    " << smooth_post_us / num_vcycles << " us" << std::endl;
        std::cout << "  Coarse smoothing:  " << smooth_coarse_us / num_vcycles << " us" << std::endl;
        std::cout << "  Residual:          " << residual_us / num_vcycles << " us" << std::endl;
        std::cout << "  Restriction:       " << restrict_us / num_vcycles << " us" << std::endl;
        std::cout << "  Prolongation:      " << prolong_us / num_vcycles << " us" << std::endl;
        std::cout << "  Memset:            " << memset_us / num_vcycles << " us" << std::endl;
        std::cout << "Convergence check:   " << convergence_check_us / num_vcycles << " us" << std::endl;
        std::cout << "Norm computation:    " << norm_us / num_vcycles << " us" << std::endl;
        std::cout << "=================================" << std::endl;
    }
    
    void reset() {
        smooth_pre_us = 0.0;
        smooth_post_us = 0.0;
        smooth_coarse_us = 0.0;
        residual_us = 0.0;
        restrict_us = 0.0;
        prolong_us = 0.0;
        norm_us = 0.0;
        convergence_check_us = 0.0;
        memset_us = 0.0;
        num_vcycles = 0;
    }
};
