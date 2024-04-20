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

	double* getPositions(void);

	double* getVelocities(void);

	unsigned int* getColors(void);

	void update(double timeDelta);

	void writecurpostofile(char* file);

	void display();

	//Particle Data Device
	double* d_positions;
	double* d_velocities;
	unsigned int* d_colors;
	unsigned char* d_particleType;

protected:

	//Kernel specs
	int blockSize;
	int gridSize;

	cudaGraphicsResource* positionResource;
	cudaGraphicsResource* colorResource;

	cudaEvent_t event;
};


#endif