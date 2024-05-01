#include "ordered.h"

/*
__global__ void update_electrons(float timeDelta, int numParticles, int numE, int numP, float* positions, float* velocities, unsigned char* particleType) {

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < numE) {
		int part_type = particleType[gid];
		float force_x = 0.0;
		float force_y = 0.0;
		float force_z = 0.0;

		for (int j = 0; j < numE + numP; j++) {
			if (gid == j) {
				continue;
			}
			float dist_x = positions[gid * 4] - positions[j * 4];
			float dist_y = positions[gid * 4 + 1] - positions[j * 4 + 1];
			float dist_z = positions[gid * 4 + 2] - positions[j * 4 + 2];
			float dist_square = (dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z);
			float dist = sqrt(dist_square);

			float force = 0.0;
			//Coulomb force
			force += (float)coulomb_scalar / dist * d_charges[part_type] * d_charges[particleType[j]];

			//Break force into components
			force_x += force * dist_x / dist;
			force_y += force * dist_y / dist;
			force_z += force * dist_z / dist;
		}

		//Update velocities
		velocities[gid * 3] += force_x * d_inv_masses[part_type] * timeDelta;
		velocities[gid * 3 + 1] += force_y * d_inv_masses[part_type] * timeDelta;
		velocities[gid * 3 + 2] += force_z * d_inv_masses[part_type] * timeDelta;

		velocities[gid * 3] *= dampingFactor;
		velocities[gid * 3 + 1] *= dampingFactor;
		velocities[gid * 3 + 2] *= dampingFactor;
	}

}

__global__ void update_protons(float timeDelta, int numParticles, int numE, int numP, int numN, float* positions, float* velocities, unsigned char* particleType) {

	int gid = numE + blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < numE + numP) {
		int part_type = particleType[gid];
		float force_x = 0.0;
		float force_y = 0.0;
		float force_z = 0.0;

		for (int j = 0; j < numE; j++) {
			float dist_x = positions[gid * 4] - positions[j * 4];
			float dist_y = positions[gid * 4 + 1] - positions[j * 4 + 1];
			float dist_z = positions[gid * 4 + 2] - positions[j * 4 + 2];
			float dist_square = (dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z);
			float dist = sqrt(dist_square);

			float force = (float)coulomb_scalar / dist * d_charges[part_type] * d_charges[particleType[j]];

			//Break force into components
			force_x += force * dist_x / dist;
			force_y += force * dist_y / dist;
			force_z += force * dist_z / dist;
		}


		for (int j = numE; j < numE + numP; j++) {
			float dist_x = positions[gid * 4] - positions[j * 4];
			float dist_y = positions[gid * 4 + 1] - positions[j * 4 + 1];
			float dist_z = positions[gid * 4 + 2] - positions[j * 4 + 2];
			float dist_square = (dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z);
			float dist = sqrt(dist_square);

			float force = (float)coulomb_scalar / dist * d_charges[part_type] * d_charges[particleType[j]];

			if (dist < yukawa_cutoff) {
				force += yukawa_scalar * exp(-dist / yukawa_radius) / dist;
			}
			else {
				force -= yukawa_scalar * exp(-dist / yukawa_radius) / dist;
			}

			//Break force into components
			force_x += force * dist_x / dist;
			force_y += force * dist_y / dist;
			force_z += force * dist_z / dist;

		}

		//Update velocities
		velocities[gid * 3] += force_x * d_inv_masses[part_type] * timeDelta;
		velocities[gid * 3 + 1] += force_y * d_inv_masses[part_type] * timeDelta;
		velocities[gid * 3 + 2] += force_z * d_inv_masses[part_type] * timeDelta;

		velocities[gid * 3] *= dampingFactor;
		velocities[gid * 3 + 1] *= dampingFactor;
		velocities[gid * 3 + 2] *= dampingFactor;
	}
}


__global__ void update_neutrons(float timeDelta, int numParticles, int numE, int numP, int numN, float* positions, float* velocities, unsigned char* particleType) {
	int gid = numE + numP + blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < numParticles) {
		int part_type = particleType[gid];
		float force_x = 0.0;
		float force_y = 0.0;
		float force_z = 0.0;

		for (int j = numE + numP; j < numParticles; j++) {
			float dist_x = positions[gid * 4] - positions[j * 4];
			float dist_y = positions[gid * 4 + 1] - positions[j * 4 + 1];
			float dist_z = positions[gid * 4 + 2] - positions[j * 4 + 2];
			float dist_square = (dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z);
			float dist = sqrt(dist_square);

			//Strong Forces
			//P-N close attraction N-N close attraction 
			float force = 0.0f;
			if (part_type != 0 && particleType[j] != 0) {
				if (dist < yukawa_cutoff) {
					force += yukawa_scalar * exp(-dist / yukawa_radius) / dist;
				}
				else {
					force -= yukawa_scalar * exp(-dist / yukawa_radius) / dist;
				}

			}

			//Break force into components
			force_x += force * dist_x / dist;
			force_y += force * dist_y / dist;
			force_z += force * dist_z / dist;
		}
		//Update velocities
		velocities[gid * 3] += force_x * d_inv_masses[part_type] * timeDelta;
		velocities[gid * 3 + 1] += force_y * d_inv_masses[part_type] * timeDelta;
		velocities[gid * 3 + 2] += force_z * d_inv_masses[part_type] * timeDelta;

		velocities[gid * 3] *= dampingFactor;
		velocities[gid * 3 + 1] *= dampingFactor;
		velocities[gid * 3 + 2] *= dampingFactor;
	}

}*/