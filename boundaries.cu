#include <boundaries.h>

void Boundaries2D::add_wall(double x1, double y1, double x2, double y2) {
	walls.push_back(x1);
	walls.push_back(y1);
	walls.push_back(x2);	
	walls.push_back(y2);
	wall_count += 1;
}

void Boundaries2D::compile_walls() {
    // Copy walls to device
    cudaMalloc(&d_walls, wall_count * sizeof(double4));
	cudaMalloc(&d_walls_int, wall_count * sizeof(int4));
    cudaMemcpy(d_walls, walls.data(), wall_count * sizeof(double4), cudaMemcpyHostToDevice);
	// Convert double walls to int4 format
	for (int i = 0; i < wall_count; ++i) {
		int4 wall_int;
		wall_int.x = static_cast<int>(walls[i * 4]);
		wall_int.y = static_cast<int>(walls[i * 4 + 1]);
		wall_int.z = static_cast<int>(walls[i * 4 + 2]);
		wall_int.w = static_cast<int>(walls[i * 4 + 3]);
		cudaMemcpy(&d_walls_int[i], &wall_int, sizeof(int4), cudaMemcpyHostToDevice);
	}
}
