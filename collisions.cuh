#pragma once


__global__ void detect_intersections(
    const double2* p0,
    const double2* p1,
    const double4* walls,
    int* collided_indices,
    int* collision_counter,
    const int num_particles,
    const int num_walls
);