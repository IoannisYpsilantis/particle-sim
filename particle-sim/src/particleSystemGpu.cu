#include "particleSystemGpu.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>


#define TILE_SIZE 128

__constant__ double inv_masses[3];
__constant__ float charges[3];

__global__ void update_naive(int numParticles, float coulomb_scalar, float yukawa_scalar, float yukawa_radius, float yukawa_cutoff, float* positions, float* velocities, unsigned char* particleType) {


}



ParticleSystemGPU::ParticleSystemGPU(int numParticles, int initMethod, int seed, Buffer* buffer) {
	p_numParticles = numParticles;
	p_buffer = buffer; //Is this right?
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



	p_buffer->mapPositions(d_positions);
	cudaMemcpy(d_positions, positions, positionElementsCount * sizeof(float), cudaMemcpyHostToDevice);
	p_buffer->unmapPositions();


	cudaMalloc(&d_velocities, velocityElementsCount * sizeof(float));
	cudaMemcpy(d_velocities, velocities, velocityElementsCount * sizeof(float), cudaMemcpyHostToDevice);

	p_buffer->mapColors(d_colors);
	cudaMemcpy(d_colors, colors, colorElementsCount * sizeof(float), cudaMemcpyHostToDevice);
	p_buffer->unmapPositions();
	
	
	cudaMalloc(&d_particleType, numParticles * sizeof(unsigned char));
	cudaMemcpy(d_particleType, particleType, numParticles * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

float* ParticleSystemGPU::getPositions() {
	p_buffer->mapPositions(d_positions);
	int numBytes = p_numParticles * 4 * sizeof(float);
	cudaMemcpy(d_positions, positions, numBytes, cudaMemcpyDeviceToHost);
	p_buffer->unmapPositions();
	return positions;
}

float* ParticleSystemGPU::getVelocities() {
	int numBytes = p_numParticles * 3 * sizeof(float);
	cudaMemcpy(d_velocities, velocities, numBytes, cudaMemcpyDeviceToHost);
	return velocities;
}

unsigned int* ParticleSystemGPU::getColors() {
	p_buffer->mapColors(d_colors);
	int numBytes = p_numParticles * 3 * sizeof(unsigned int);
	cudaMemcpy(d_colors, colors, numBytes, cudaMemcpyDeviceToHost);
	p_buffer->unmapColors();
	return colors;
}



void ParticleSystemGPU::update(float timeDelta) {
	update_naive<<<gridSize, blockSize>>>(p_numParticles, coulomb_scalar, yukawa_scalar, yukawa_radius, yukawa_cutoff, positions, velocities, particleType);
}



void ParticleSystemGPU::writecurpostofile(char* file) {
	
}

	


ParticleSystemGPU::~ParticleSystemGPU() {
	p_numParticles = 0;
	delete[] positions;
	delete[] colors;
	delete[] velocities;
	delete[] particleType;

	//VBO will handle positions and colors buffers.
	cudaFree(d_velocities);
	cudaFree(d_particleType);
}