#include "particles.cuh"
#include "iostream"
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <ratio>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

__global__ void decrese_last_alive(int* d_last_alive, int* size) {
    if (threadIdx.x == 0) {
        (*d_last_alive)-= *size;
    }
}

__global__ void reset_counters(int* count_head, int* count_tail) {
    if (threadIdx.x == 0) {
        *count_head = 0;
        *count_tail = 0;
    }
}

__global__ void mark_collided(
    const int* collided_indices,
    int num_remove,
    int stamp,
    int* collision_flags
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_remove) {
        int idx = collided_indices[tid];
        collision_flags[idx] = stamp;
    }
}


__global__ void mark_and_collect(
    const int* collided_indices,
    int num_remove,
    int last_alive,
    int head_limit,
    const int* collision_flags,
    int* d_remove_in_head,
    int* d_stay_in_tail,
    int* d_count_head,
    int* d_count_tail,
    int stamp
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_remove) return;

    int idx = collided_indices[tid];
    
    // Verify that this particle was actually marked in this step
    if (collision_flags[idx] != stamp) return;

    if (idx < head_limit) {
        // belongs to head, must be removed
        int out_idx = atomicAdd(d_count_head, 1);
        d_remove_in_head[out_idx] = idx;
    } else {
        // belongs to tail but survived, can be swapped with head
        int out_idx = atomicAdd(d_count_tail, 1);
        d_stay_in_tail[out_idx] = idx;
    }
}

__global__ void collect_tail_survivors_stamp(
    int head_limit,
    int last_alive,
    const int* __restrict__ flags,
    int stamp,
    int* __restrict__ stay_in_tail,
    int* __restrict__ count_tail
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = head_limit + tid;
    if (i >= last_alive) return;

    if (flags[i] != stamp) {
        int pos = atomicAdd(count_tail, 1);
        stay_in_tail[pos] = i;
    }
}

// C) Bijective copy (no swap) with double2
__global__ void bijective_copy_kernel(
    const int* __restrict__ remove_in_head,  // dst indices in [0, head_limit)
    const int* __restrict__ stay_in_tail,    // src indices in [head_limit, last_alive)
    const int* __restrict__ count_head,     // number of particles to remove in head
    const int* __restrict__ count_tail,     // number of particles to stay in tail
    double2* __restrict__ r,                 // position
    double2* __restrict__ v,                 // velocity
    int* __restrict__ type                   // any extra AoS/SoA field
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int count = min(*count_head, *count_tail); // number of particles to remove in head
    
    if (tid >= count) return;
    
    const int dst = remove_in_head[tid];
    const int src = stay_in_tail[tid];

    // 16-byte vector copies
    r[dst] = r[src];
    v[dst] = v[src];
    type[dst] = type[src];
}

Particles2D::Particles2D(const int num_particles){
    max_particles = num_particles;
    last_alive = 0;
    h_current_stamp = 1;
    if (cudaMalloc(&d_last_alive, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&r, num_particles * sizeof(double2)) != cudaSuccess ||
        cudaMalloc(&r_old, num_particles * sizeof(double2)) != cudaSuccess ||
        cudaMalloc(&v, num_particles * sizeof(double2)) != cudaSuccess ||
        cudaMalloc(&v_abs, num_particles * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&energy, num_particles * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&crs, num_particles * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&type, num_particles * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&q, species_num * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&m, species_num * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&collided_indices, num_particles * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&collision_counter, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&swap_counter, sizeof(int)) != cudaSuccess||
        cudaMalloc(&d_collision_flags, num_particles * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_count_head, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_count_tail, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_remove_in_head, num_particles * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_stay_in_tail, num_particles * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&ionization_flags, num_particles * sizeof(int)) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaMemset(d_last_alive, 0, sizeof(int));
    cudaMemset(d_collision_flags, 0, num_particles * sizeof(int));
    cudaMemset(d_count_head, 0, sizeof(int));
    cudaMemset(d_count_tail, 0, sizeof(int));
    cudaMemset(d_remove_in_head, 0, num_particles * sizeof(int));
    cudaMemset(d_stay_in_tail, 0, num_particles * sizeof(int));

    double h_q = 1.0f;
    double h_m = 1.0f;
    cudaMemcpy(q, &h_q, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m, &h_m, sizeof(double), cudaMemcpyHostToDevice);
}


Particles2D::~Particles2D(){
    cudaFree(r);
    cudaFree(r_old);
    cudaFree(v);
    cudaFree(type);
    cudaFree(q);
    cudaFree(m);
    cudaFree(v_abs);
    cudaFree(energy);
    cudaFree(crs);
    cudaFree(collided_indices);
    cudaFree(collision_counter);
    cudaFree(d_last_alive);
    cudaFree(d_collision_flags);
    cudaFree(d_count_head);
    cudaFree(d_count_tail);
    cudaFree(d_remove_in_head);
    cudaFree(d_stay_in_tail);
}


void Particles2D::add_species(double new_m, double new_q){

    double* new_q_ptr;
    double* new_m_ptr;
    cudaMalloc(&new_q_ptr, (species_num + 1) * sizeof(double));
    cudaMalloc(&new_m_ptr, (species_num + 1) * sizeof(double));
    cudaMemcpy(new_q_ptr, q, species_num * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_m_ptr, m, species_num * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_q_ptr + species_num, &new_q, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(new_m_ptr + species_num, &new_m, sizeof(double), cudaMemcpyHostToDevice);
    cudaFree(q);
    cudaFree(m);
    q = new_q_ptr;
    m = new_m_ptr;
    species_num++;

}


void Particles2D::change_species(int specise, double new_m, double new_q){
    cudaMemcpy(q + specise, &new_q, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m + specise, &new_m, sizeof(double), cudaMemcpyHostToDevice);
}



void Particles2D::debugParticles(int num_to_print) {
    std::vector<double2> h_pos(std::min(last_alive, num_to_print));
    std::vector<double2> h_vel(std::min(last_alive, num_to_print));

    int particles_to_check = std::min(last_alive, num_to_print);
    cudaMemcpy(h_pos.data(), r, particles_to_check * sizeof(double2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel.data(), v, particles_to_check * sizeof(double2), cudaMemcpyDeviceToHost);

    std::cout << "Particle debug - first " << particles_to_check << " particles:" << std::endl;
    for (int i = 0; i < particles_to_check; i++) {
        std::cout << "P" << i << ": pos=(" << h_pos[i].x << "," << h_pos[i].y
            << ") vel=(" << h_vel[i].x << "," << h_vel[i].y << ")" << std::endl;
    }
}



void Particles2D::removeCollidedParticles() {

    int h_num_remove;
    cudaMemcpy(&h_num_remove, collision_counter, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_num_remove == 0) return;

    int head_limit = last_alive - h_num_remove;

    int threads = 256;
    int blocks = (h_num_remove + threads - 1) / threads;

    // Step 1: mark all collided with current stamp
    mark_collided<<<blocks, threads>>>(collided_indices,
                                       h_num_remove,
                                       h_current_stamp,
                                       d_collision_flags);

    // Step 2: classify into head/tail
    reset_counters<<<1, 1>>>(d_count_head, d_count_tail);

    mark_and_collect<<<blocks, threads>>>(
        collided_indices,
        h_num_remove,
        last_alive,
        head_limit,
        d_collision_flags,
        d_remove_in_head,
        d_stay_in_tail,
        d_count_head,
        d_count_tail,
        h_current_stamp
    );

    collect_tail_survivors_stamp<<<((last_alive-head_limit)+255)/256, threads>>>(   // Use the updated threads variable
    head_limit, last_alive, d_collision_flags, h_current_stamp,
    d_stay_in_tail, d_count_tail);

    bijective_copy_kernel<<<(blocks+255)/256, threads>>>(   // Use the updated threads variable
        d_remove_in_head, d_stay_in_tail, d_count_head, d_count_tail, r, v, type);


    last_alive -= h_num_remove;
    h_current_stamp++;   // increment for next step

}