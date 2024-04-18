#include "particleSystemGpu.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>


#define TILE_SIZE 128

__constant__ double inv_masses[3];
__constant__ float charges[3];


__global__ void update_naive(float timeDelta, int numParticles, float coulomb_scalar, float yukawa_scalar, float yukawa_radius, float yukawa_cutoff, float* positions, float* velocities, unsigned char* particleType) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int part_type = particleType[gid];
	double force_x = 0.0f;
	double force_y = 0.0f;
	double force_z = 0.0f;
	for (int j = 0; j < numParticles; j++) {
		float dist_square = (positions[gid] - positions[j]) * (positions[gid] - positions[j]) + (positions[gid] - positions[j + 1]) * (positions[gid] - positions[j + 1]);
		float dist = sqrt(dist_square);
		if (gid == j || dist < yukawa_cutoff) {
			continue;
		}

		double force = (double)coulomb_scalar / dist_square * charges[part_type] * charges[particleType[j]];
		double dist_x = (double)positions[gid] - positions[j];
		double dist_y = (double)positions[gid + 1] - positions[j + 1];
		force_x += force * dist_x / dist;
		force_y += force * dist_y / dist;

		//Strong Forces
		//P-N close attraction N-N close attraction 
		if (part_type != 0 && particleType[j] != 0) {
			force = yukawa_scalar * exp(dist / yukawa_radius) / dist;
			force_x += force * dist_x / dist;
			force_y += force * dist_y / dist;
		}

		//Update velocities
		velocities[gid] += force_x * inv_masses[part_type] * 1e-9 * timeDelta;
		velocities[gid + 1] += force_y * inv_masses[part_type] * 1e-9 * timeDelta;
		velocities[gid + 2] += force_z * inv_masses[part_type] * 1e-9 * timeDelta;
		
		//Update positions from velocities
		positions[gid * 4] += velocities[gid * 3];
		if (abs(positions[gid * 4]) > 1) {
			velocities[gid * 3] = -1 * velocities[gid * 3];
		}
		positions[gid * 4 + 1] += velocities[gid * 3 + 1];
		if (abs(positions[gid * 4 + 1]) > 1) {
			velocities[gid * 3 + 1] = -1 * velocities[gid * 3 + 1];
		}
		positions[gid * 4 + 2] += velocities[gid * 3 + 2];
		if (abs(positions[gid * 4 + 2]) > 1) {
			velocities[gid * 3 + 2] = -1 * velocities[gid * 3 + 2];
		}
	}


}



ParticleSystemGPU::ParticleSystemGPU(int numParticles, int initMethod, int seed, bool render) {
	p_numParticles = numParticles;
	p_render = render;
	blockSize = TILE_SIZE;
	gridSize = (int)ceil((float)numParticles / (float)TILE_SIZE);

	


	// Initialize Positions array
	int positionElementsCount = 4 * numParticles;
	positions = new float[positionElementsCount];
	//memset(positions, 0, positionElementsCount);

	// Initialize Colors array
	int colorElementsCount = 3 * numParticles;
	colors = new unsigned int[colorElementsCount];
	//memset(colors, 0, colorElementsCount);

	int velocityElementsCount = 3 * numParticles;
	velocities = new float[velocityElementsCount];

	// Initialize Particle Type array
	particleType = new unsigned char[numParticles];

	coulomb_scalar = 2.310272969e-4; //N*nanometers^2
	yukawa_scalar = 1.9692204e-3;    //Experimentally obtained
	yukawa_radius = 1.4e-3;			 //Radius of strength.
	yukawa_cutoff = 1e-3;          //Sweet spot. (Strong force likes to be between 0.8 and 1.4 fm.

	// Circular initialization
	if (initMethod == 0) {
		for (unsigned int i = 0; i < numParticles; i++) {
			float theta = (float)((numParticles - 1 - i) / (float)numParticles * 2.0 * 3.1415); // Ensure floating-point division
			positions[i * 4] = (float)cos(theta);
			positions[i * 4 + 1] = (float)sin(theta);
			positions[i * 4 + 2] = 1.0f;
			positions[i * 4 + 3] = 1.0f; // This will always stay as 1, it will be used for mapping 3D to 2D space

			colors[i * 3] = i % 255;
			colors[i * 3 + 1] = 255 - (i % 255);
			colors[i * 3 + 2] = 55;
		}

	}
	//Read from a file
	else if (initMethod == 1) {

	}
	// Random initialization in 3 dimensions
	else if (initMethod == 2) {
		if (seed != -1) {
			srand(seed);
		}
		for (unsigned int i = 0; i < numParticles; i++) {
			// Randomly initialize position in range [-1,1)
			positions[i * 4] = ((float)(rand() % 2000) - 1000.0) / 1000.0;
			positions[i * 4 + 1] = ((float)(rand() % 2000) - 1000.0) / 1000.0;
			positions[i * 4 + 2] = ((float)(rand() % 2000) - 1000.0) / 1000.0;
			positions[i * 4 + 3] = 1.0f; // This will always stay as 1, it will be used for mapping 3D to 2D space

			// Randomly initializes velocity in range [-0.0025,0.0025)
			velocities[i * 3] = ((float)(rand() % 500) - 250.0) / 100000.0;
			velocities[i * 3 + 1] = ((float)(rand() % 500) - 250.0) / 100000.0;
			velocities[i * 3 + 2] = ((float)(rand() % 500) - 250.0) / 100000.0;

			// Generates random number (either 0, 1, 2) from uniform dist
			//particleType[i] = rand() % 3 % 2; 
			particleType[i] = rand() % 3;

			// Sets color based on particle type
			if (particleType[i] == 0) { // If Electron
				colors[i * 3] = 0;
				colors[i * 3 + 1] = 180;
				colors[i * 3 + 2] = 255;

			}
			else if (particleType[i] == 1) { // If Proton
				colors[i * 3] = 255;
				colors[i * 3 + 1] = 0;
				colors[i * 3 + 2] = 0;
			}
			else {
				colors[i * 3] = 255; //Else neutron
				colors[i * 3 + 1] = 0;
				colors[i * 3 + 2] = 180;

			}
		}
	}
	//Error bad method
	else {

	}

	//Initialize device

	double inv_mass[] = { 1.09776e30, 5.978638e26, 5.978638e26 };
	float charge[] = { -1, 1, 0 };

	cudaMemcpyToSymbol(inv_masses, inv_mass, 3 * sizeof(double));
	cudaMemcpyToSymbol(charges, charge, 3 * sizeof(float));


	if (!render) {
		cudaMalloc(&d_positions, positionElementsCount * sizeof(float));
		cudaMemcpy(d_positions, velocities, velocityElementsCount * sizeof(float), cudaMemcpyHostToDevice);
	}
	//p_buffer->mapPositions(d_positions);
	//cudaMemcpy(d_positions, positions, positionElementsCount * sizeof(float), cudaMemcpyHostToDevice);
	//p_buffer->unmapPositions();


	cudaMalloc(&d_velocities, velocityElementsCount * sizeof(float));
	cudaMemcpy(d_velocities, velocities, velocityElementsCount * sizeof(float), cudaMemcpyHostToDevice);

	if (!render) {
		cudaMalloc(&d_colors, colorElementsCount * sizeof(unsigned int));
		cudaMemcpy(d_colors, colors, colorElementsCount * sizeof(unsigned int), cudaMemcpyHostToDevice);
	}
	//p_buffer->mapColors(d_colors);
	//cudaMemcpy(d_colors, colors, colorElementsCount * sizeof(float), cudaMemcpyHostToDevice);
	//p_buffer->unmapPositions();
	
	
	cudaMalloc(&d_particleType, numParticles * sizeof(unsigned char));
	cudaMemcpy(d_particleType, particleType, numParticles * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

//This should be run before any other functions. (The construction is dependent on this running.
void ParticleSystemGPU::assignBuffer(Buffer* buffer) {
	p_buffer = buffer;
}

float* ParticleSystemGPU::getPositions() {
	if (p_render) {
		p_buffer->mapPositions(d_positions);
	}
	
	int numBytes = p_numParticles * 4 * sizeof(float);
	cudaMemcpy(d_positions, positions, numBytes, cudaMemcpyDeviceToHost);
	if (p_render) {
		p_buffer->unmapPositions();
	}
	
	return positions;
}

float* ParticleSystemGPU::getVelocities() {
	int numBytes = p_numParticles * 3 * sizeof(float);
	cudaMemcpy(d_velocities, velocities, numBytes, cudaMemcpyDeviceToHost);
	return velocities;
}

unsigned int* ParticleSystemGPU::getColors() {
	if (p_render) {
		p_buffer->mapColors(d_colors);
	}
	
	int numBytes = p_numParticles * 3 * sizeof(unsigned int);
	cudaMemcpy(d_colors, colors, numBytes, cudaMemcpyDeviceToHost);
	if (p_render) {
		p_buffer->unmapColors();
	}
	
	return colors;
}



void ParticleSystemGPU::update(float timeDelta) {
	update_naive<<<gridSize, blockSize>>>(timeDelta, p_numParticles, coulomb_scalar, yukawa_scalar, yukawa_radius, yukawa_cutoff, positions, velocities, particleType);
}



void ParticleSystemGPU::writecurpostofile(char* file) {
	getPositions();
	std::ofstream outfile(file);

	if (outfile.is_open()) {
		for (int i = 0; i < p_numParticles; i++) {
			outfile << positions[i * 4] << " ";
			outfile << positions[i * 4 + 1] << " ";
			outfile << positions[i * 4 + 2] << " ";
			outfile << positions[i * 4 + 3] << "\n";
		}
	}
	else {
		std::cerr << "Unable to open file: " << file << std::endl;
	}
}

	


ParticleSystemGPU::~ParticleSystemGPU() {
	p_numParticles = 0;
	p_render = 0;
	delete[] positions;
	delete[] colors;
	delete[] velocities;
	delete[] particleType;

	//VBO will handle positions and colors buffers if we rendered.
	cudaFree(d_velocities);
	cudaFree(d_particleType);
	if (!p_render) {
		cudaFree(d_positions);
		cudaFree(d_colors);
	}
	
}