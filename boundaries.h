#pragma once
#include <vector>
#include <cuda_runtime.h>

using namespace std;

class Boundaries2D{
    public:
    Boundaries2D(int width, int height, int nx, int ny) 
        : width(width), height(height), nx(nx), ny(ny) {
    }
    void add_wall(double x1, double y1, double x2, double y2);
    int wall_count = 0;
    vector<double> walls; // Stores walls as [x1, y1, x2, y2, ...]
    double4* d_walls = nullptr; // Device pointer for walls
    int4* d_walls_int = nullptr; // Device pointer for walls in integer format
    void compile_walls();

    private:
    int width, height;
    int nx, ny; // Grid dimensions
    
};