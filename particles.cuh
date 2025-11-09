#pragma once

class Particles2D {

public:
	int max_particles;
	double2* r;
	double2* r_old; // old positions for velocity calculation
	double2* v;
	double* v_abs;
	double* energy;
	double* crs; // cross section
	int* type;
	double* q;
	double* m;
	int* collided_indices; // indices of particles that collided with walls
	int* collision_counter; // counter for collisions
	int* swap_counter; // counter for swaps
	int* d_last_alive;
	int last_alive;
	int* d_collision_flags;
	int* d_count_head;
	int* d_count_tail;
	int* d_remove_in_head; // indices of particles to remove in the head
	int* d_stay_in_tail; // indices of particles to stay in the tail
	int h_current_stamp; // current stamp for collision detection
	int* ionization_flags; // flags for ionization events

	Particles2D(const int num_particles);
	~Particles2D();
	void add_species(double m, double q);
	void change_species(int species, double m, double q);
	void debugParticles(int num_to_print = 5);
	void removeCollidedParticles();

private:
	int species_num = 1;

};

