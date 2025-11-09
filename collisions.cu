#include <collisions.cuh>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>


__global__
void detect_intersections(
    const double2* p0,
    const double2* p1,
    const double4* walls,
    int* collided_indices,
    int* collision_counter,
    const int num_particles,
    const int num_walls
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_particles) return;

    double x1 = p0[i].x;
    double y1 = p0[i].y;
    double x2 = p1[i].x;
    double y2 = p1[i].y;

    bool intersects = false;
    const double EPSILON = 1e-10;
    
    for (int j = 0; j < num_walls && !intersects; ++j) {
        double x3 = walls[j].x;
        double y3 = walls[j].y;
        double x4 = walls[j].z;
        double y4 = walls[j].w;

        double denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1);
        
        if (fabs(denom) < EPSILON) continue; // Parallel lines
        
        double ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom;
        double ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom;
        
        if (ua >= 0.0 && ua <= 1.0 && ub >= 0.0 && ub <= 1.0) {
            intersects = true;
        }
    }
    
    if (intersects) {
        // Atomically increment and get write position
        int write_idx = atomicAdd(collision_counter, 1);
        // Add bounds check to prevent buffer overflow
        if (write_idx < num_particles) {
            collided_indices[write_idx] = i;  // Store segment index that collided
        }
    }
}



__global__
void detect_intersections_mask(
    const double* p0,
    const double* p1,
    const double* walls,
    int* collision_mask,
    const int num_segments,
    const int num_walls
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_segments) return;
    
    double x1 = p0[2*i];
    double y1 = p0[2*i + 1];
    double x2 = p1[2*i];
    double y2 = p1[2*i + 1];
    
    bool intersects = false;
    const double EPSILON = 1e-10;
    
    for (int j = 0; j < num_walls && !intersects; ++j) {
        double x3 = walls[4*j];
        double y3 = walls[4*j + 1];
        double x4 = walls[4*j + 2];
        double y4 = walls[4*j + 3];
        
        double denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1);
        if (fabs(denom) < EPSILON) continue;
        
        double ua = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom;
        double ub = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom;
        
        if (ua >= 0.0 && ua <= 1.0 && ub >= 0.0 && ub <= 1.0) {
            collision_mask[i] = i + 1;
        }
    }
    
}



