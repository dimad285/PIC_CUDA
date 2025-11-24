#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cusparse.h>
#include <particles.cuh>
#include <grid.cuh>
#include <chrono>
#include <random>
#include <Multigrid.hpp>
#include <mcc.hpp>
#include <render.h>
#include <diagnostics.h>
#include <boundaries.h>
#include "collisions.cuh"


void initializeParticles(double* h_positions, double* h_velocities, int num_particles) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0.2, 0.8);
    std::uniform_real_distribution<double> vel_dist(-0.01, 0.01);

    for (int i = 0; i < num_particles; ++i) {
        // Distribute particles randomly in the domain
        h_positions[2 * i] = pos_dist(gen);
        h_positions[2 * i + 1] = pos_dist(gen);

        // Random velocities
        h_velocities[2 * i] = vel_dist(gen);
        h_velocities[2 * i + 1] = vel_dist(gen);
    }
}

// Structure to hold timing information for each simulation step


void updateSimulationCUDA(Particles2D& particles, Grid2D& grid, Boundaries2D& boundaries, int& step, double dt, int steps, SimulationTiming& timing) {
    auto total_start = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int particleBlocks = (particles.last_alive + threadsPerBlock - 1) / threadsPerBlock;

    dim3 block_dim(16, 16);
    dim3 grid_dim((grid.m + block_dim.x - 1) / block_dim.x,
        (grid.n + block_dim.y - 1) / block_dim.y);

    // Reset timing accumulators
    timing.coordinate_update_us = 0.0;
    timing.density_reset_us = 0.0;
    timing.density_update_us = 0.0;
    timing.solver_us = 0.0;
    timing.electric_field_us = 0.0;
    timing.velocity_update_us = 0.0;
    timing.mcc_us = 0.0;
    timing.synchronize_us = 0.0;
    timing.total_us = 0.0;
    timing.collision_detection_us = 0.0;
    timing.particle_removal = 0.0;

    for (int i = 0; i < steps; i++) {
        // Time coordinate update
        auto step_start = std::chrono::high_resolution_clock::now();

        update_coordinates << <particleBlocks, threadsPerBlock >> > (
            particles.r, particles.r_old, particles.v, particles.last_alive, dt);
        //cudaDeviceSynchronize(); // Sync for accurate timing
        auto step_end = std::chrono::high_resolution_clock::now();
        timing.coordinate_update_us += std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();

        step_start = std::chrono::high_resolution_clock::now();
        // Detect collisions with walls
        cudaMemset(particles.collision_counter, 0, sizeof(int)); // Reset collision counter
        cudaMemset(particles.collided_indices, -1, particles.last_alive * sizeof(int)); // Reset collided indices
        detect_intersections << <particleBlocks, threadsPerBlock >> > (
            particles.r_old, particles.r, boundaries.d_walls, particles.collided_indices, particles.collision_counter,
            particles.last_alive, boundaries.wall_count);
        // Time density reset

        step_end = std::chrono::high_resolution_clock::now();
        timing.collision_detection_us += std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();

        step_start = std::chrono::high_resolution_clock::now();

        particles.removeCollidedParticles();

        step_end = std::chrono::high_resolution_clock::now();
        auto remove_time = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();

        timing.particle_removal += remove_time;

        step_start = std::chrono::high_resolution_clock::now();

        cudaMemset(grid.rho, 0.0, grid.m * grid.n * sizeof(double));
        //cudaDeviceSynchronize(); // Sync for accurate timing

        step_end = std::chrono::high_resolution_clock::now();
        timing.density_reset_us += std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();

        // Time density update
        step_start = std::chrono::high_resolution_clock::now();

        updateDensity << <particleBlocks, threadsPerBlock >> > (
            grid.rho, particles.r, particles.q, particles.type,
            grid.m, grid.n, grid.dx, grid.dy, particles.last_alive);
        //cudaDeviceSynchronize(); // Sync for accurate timing

        step_end = std::chrono::high_resolution_clock::now();
        timing.density_update_us += std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();


        // Add timing for additional physics steps when enabled
        step_start = std::chrono::high_resolution_clock::now();
        solve_multigrid(grid.phi, grid.rho, grid.m, grid.n, 5, grid.dx);
        //solver.solve(grid.phi, grid.rho);
        step_end = std::chrono::high_resolution_clock::now();
        timing.solver_us += std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();

        step_start = std::chrono::high_resolution_clock::now();
        computeElectricField << <grid_dim, block_dim >> > (grid.E, grid.phi, grid.m, grid.n, grid.dx, grid.dy);
        step_end = std::chrono::high_resolution_clock::now();
        timing.electric_field_us += std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();


        step_start = std::chrono::high_resolution_clock::now();
        updateParticleVelocities << <particleBlocks, threadsPerBlock >> > (
            particles.v, grid.E, particles.r,
            particles.type, particles.q, particles.m, particles.last_alive, grid.dx, grid.dy, grid.m, grid.n, dt);
        step_end = std::chrono::high_resolution_clock::now();
        timing.velocity_update_us += std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();

        //step_start = std::chrono::high_resolution_clock::now();
        //elasticCollisions(particles, 1, dt, crossections);
        //step_end = std::chrono::high_resolution_clock::now();
        //timing.mcc_us += std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();
    }

    // Final synchronization timing
    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
}




// --- Main Function ---
int main() {
    // --- Simulation Parameters ---
    const int MAX_PARTICLES = 1100000;  // Reduced for better performance
    int num_particles_sim = 1000000;     // Start with fewer particles
    const double DELTA_TIME = 0.01;   // Larger time step
    const int UPDATE_STEPS_PER_FRAME = 1;  // Fewer steps per frame for real-time
    int step = 1;

    // Grid parameters
    int grid_m = 129;
    int grid_n = 129;
    double domain_X = 1.0;
    double domain_Y = 1.0;
    double grid_dx = domain_X / (grid_m - 1);
    double grid_dy = domain_Y / (grid_n - 1);

    int mg_levels = 4;

    // --- Initialization ---
    std::cout << "Initializing GLFW and GLEW..." << std::endl;
    if (!initializeGLFW_GLEW()) {
        return -1;
    }
    std::cout << "GLFW and GLEW initialized successfully." << std::endl;

    std::cout << "Creating shader program..." << std::endl;
    GLuint shaderProgram = createShaderProgram();
    if (shaderProgram == 0) {
        cleanupGraphics(0, 0);
        return -1;
    }
    std::cout << "Shader program created." << std::endl;

    std::cout << "Setting up particle rendering..." << std::endl;
    GLuint particleVAO = setupParticleRendering(MAX_PARTICLES, shaderProgram);
    std::cout << "Particle rendering setup complete." << std::endl;
    std::cout << "Setting up line rendering..." << std::endl;
    setupLineRendering();
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    std::cout << "Allocating simulation objects..." << std::endl;
    Particles2D particles(MAX_PARTICLES);
    particles.add_species(1.0, -1.0);

    Grid2D grid(grid_m, grid_n, domain_X, domain_Y);
    
    Boundaries2D boundaries(domain_X, domain_Y, grid_m, grid_n);
    boundaries.add_wall(domain_X / 4, domain_Y / 4, domain_X / 4, domain_Y * 3 / 4);  // Left wall
    boundaries.add_wall(domain_X * 3 / 4, domain_Y / 4, domain_X * 3 / 4, domain_Y * 3 / 4); // Right wall
    boundaries.compile_walls();

    for (int i = 0; i < boundaries.wall_count; ++i) {
        float x1 = static_cast<float>(boundaries.walls[i * 4]);
        float y1 = static_cast<float>(boundaries.walls[i * 4 + 1]);
        float x2 = static_cast<float>(boundaries.walls[i * 4 + 2]);
        float y2 = static_cast<float>(boundaries.walls[i * 4 + 3]);
        addLineSegment(x1, y1, x2, y2);
    }

    updateLineBuffer();
    //CudaPoissonSolver solver(grid_m, grid_n, grid_dx, grid_dy, 1e-5, 1);
    //MultigridSolver solver(grid_m, grid_n, mg_levels, grid_dx, 1.25);
    /*
    cross_sections crossections;

    std::pair<std::vector<double>, std::vector<double>> css = read_cross_section("csf/Ar.txt");
    cudaMalloc(&crossections.energies, css.first.size() * sizeof(double));
    cudaMalloc(&crossections.sigmas, css.second.size() * sizeof(double));

    cudaMemcpy(crossections.energies, css.first.data(),
        css.first.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(crossections.sigmas, css.second.data(),
        css.second.size() * sizeof(double), cudaMemcpyHostToDevice);

    crossections.size = css.first.size();
    */

    std::cout << "Simulation objects allocated." << std::endl;

    std::cout << "Initializing particles on host..." << std::endl;
    std::vector<double> h_positions(2 * num_particles_sim);
    std::vector<double> h_velocities(2 * num_particles_sim);
    initializeParticles(h_positions.data(), h_velocities.data(), num_particles_sim);
    std::cout << "Particles initialized on host." << std::endl;

    std::cout << "Copying particle data to device..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(particles.r, h_positions.data(),
        num_particles_sim * sizeof(double2), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(particles.r_old, h_positions.data(),
        num_particles_sim * sizeof(double2), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(particles.v, h_velocities.data(),
        num_particles_sim * sizeof(double2), cudaMemcpyHostToDevice));
    particles.last_alive = num_particles_sim;
    std::cout << "Particle data copied to device." << std::endl;

    std::cout << "Starting simulation main loop..." << std::endl;

    auto last_debug_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    SimulationTiming frame_timing;
    SimulationTiming accumulated_timing = {};
    double accumulated_vbo_time = 0.0;
    double accumulated_render_time = 0.0;
    bool show_detailed_timing = false;
    int timing_frames = 0;

    // --- Main Loop ---
    while (!glfwWindowShouldClose(g_window)) {
        glfwPollEvents();

        glClearColor(0.0f, 0.0f, 0.1f, 1.0f); // Dark blue background
        glClear(GL_COLOR_BUFFER_BIT);

        auto frame_start = std::chrono::high_resolution_clock::now();

        // Update simulation with detailed timing
        updateSimulationCUDA(particles, grid, boundaries, step, DELTA_TIME, UPDATE_STEPS_PER_FRAME, frame_timing);

        // Time VBO update
        double vbo_time = updateVBO(particles);

        // Time rendering
        auto render_start = std::chrono::high_resolution_clock::now();
        renderParticles(shaderProgram, particleVAO, particles.last_alive);
        renderLineSegments(1.0f, 0.0f, 0.0f, 1.0f); // Red color for walls
        auto render_end = std::chrono::high_resolution_clock::now();
        double render_time = std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start).count();

        glfwSwapBuffers(g_window);

        auto frame_end = std::chrono::high_resolution_clock::now();
        auto total_frame_time = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count();

        frame_count++;
        timing_frames++;

        // Accumulate timing data
        accumulated_timing.coordinate_update_us += frame_timing.coordinate_update_us;
        accumulated_timing.collision_detection_us += frame_timing.collision_detection_us;
        accumulated_timing.particle_removal += frame_timing.particle_removal;
        accumulated_timing.density_reset_us += frame_timing.density_reset_us;
        accumulated_timing.density_update_us += frame_timing.density_update_us;
        accumulated_timing.solver_us += frame_timing.solver_us;
        accumulated_timing.electric_field_us += frame_timing.electric_field_us;
        accumulated_timing.velocity_update_us += frame_timing.velocity_update_us;
        accumulated_timing.synchronize_us += frame_timing.synchronize_us;
        accumulated_timing.mcc_us += frame_timing.mcc_us;
        accumulated_timing.total_us += frame_timing.total_us;
        accumulated_vbo_time += vbo_time;
        accumulated_render_time += render_time;

        // Debug output every 2 seconds
        if (std::chrono::duration_cast<std::chrono::seconds>(frame_end - last_debug_time).count() >= 2) {
            double avg_fps = frame_count / 2.0;
            double avg_frame_time_ms = total_frame_time / 1000.0;

            std::cout << "\n=== FRAME PERFORMANCE SUMMARY ===" << std::endl;
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "FPS: " << avg_fps << ", Avg Frame Time: " << avg_frame_time_ms << " ms" << std::endl;

            // Show breakdown of frame time
            std::cout << std::setprecision(2);
            std::cout << "Frame Time Breakdown (average µs):" << std::endl;
            std::cout << "  Simulation:  " << accumulated_timing.total_us / timing_frames << " µs" << std::endl;
            std::cout << "  VBO Update:  " << accumulated_vbo_time / timing_frames << " µs" << std::endl;
            std::cout << "  Rendering:   " << accumulated_render_time / timing_frames << " µs" << std::endl;

            // Show detailed simulation timing every 4th report (every 8 seconds)
            if (show_detailed_timing) {
                // Average the accumulated timing
                SimulationTiming avg_timing = accumulated_timing;
                avg_timing.coordinate_update_us /= timing_frames;
                avg_timing.collision_detection_us /= timing_frames;
                avg_timing.particle_removal /= timing_frames;
                avg_timing.density_reset_us /= timing_frames;
                avg_timing.density_update_us /= timing_frames;
                avg_timing.solver_us /= timing_frames;
                avg_timing.electric_field_us /= timing_frames;
                avg_timing.velocity_update_us /= timing_frames;
                avg_timing.synchronize_us /= timing_frames;
                avg_timing.mcc_us /= timing_frames;
                avg_timing.total_us /= timing_frames;

                printDetailedTiming(avg_timing, UPDATE_STEPS_PER_FRAME, particles.last_alive);
                //debugParticles(particles.r, particles.v, particles.last_alive, 3);
                //debugGrid(grid);
            }

            // Reset counters
            frame_count = 0;
            timing_frames = 0;
            accumulated_timing = {};
            accumulated_vbo_time = 0.0;
            accumulated_render_time = 0.0;
            last_debug_time = frame_end;
            show_detailed_timing = !show_detailed_timing; // Toggle detailed timing
        }
    }

    // --- Cleanup ---
    std::cout << "Simulation finished. Cleaning up resources..." << std::endl;
    cleanupGraphics(shaderProgram, particleVAO);
    std::cout << "Cleanup complete. Exiting." << std::endl;

    return 0;
}