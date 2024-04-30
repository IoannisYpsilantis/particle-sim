#ifndef PARTICLESYSTEMGPU_H
#define PARTICLESYSTEMGPU_H

// Standard Library Includes
#include <iostream>
#include <fstream>

// Program Includes
#include "particleSystem.h"

// Visualization Includes
#include <glad/glad.h>

// GPU Library Includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

class ParticleSystemGPU : public ParticleSystem
{
public:
	ParticleSystemGPU(int numParticles, int init_method, int seed);

	~ParticleSystemGPU(void);

	float* getPositions(void);

	float* getVelocities(void);

	unsigned int* getColors(void);

	void update(float timeDelta);

	void flip();

	void writecurpostofile(char* file, int steps, float milliseconds);

	void display();

	//Particle Data Device
	float* d_positions;
	float* d_positions2;
	float* d_velocities;
	unsigned int* d_colors;
	unsigned char* d_particleType;

protected:

	//Kernel specs
	int blockSize;
	int gridSize;

	cudaGraphicsResource* positionResource;
	cudaGraphicsResource* colorResource;

	cudaEvent_t event;

	bool buf;
	float* src;
	float* dst;
};


#endif