#ifndef PARTICLESYSTEMGPU_H
#define PARTICLESYSTEMGPU_H

#include "particleSystem.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <fstream>

class ParticleSystemGPU : public ParticleSystem
{
public:
	ParticleSystemGPU(int numParticles, int init_method, int seed, bool render);

	~ParticleSystemGPU(void);

	float* getPositions(void);

	float* getVelocities(void);

	unsigned int* getColors(void);

	void update(float timeDelta);

	void writecurpostofile(char* file);

	void display();

	//Particle Data Device
	float* d_positions;
	float* d_velocities;
	unsigned int* d_colors;
	unsigned char* d_particleType;

protected:

	int p_numParticles;
	bool p_render;

	//Kernel specs
	int blockSize;
	int gridSize;

	//Particle Data Host
	float* positions; // 1D Array containing spacial data of each particle (positionElementsCount * numParticles)
	float* velocities; // 1D Array containing velocity data of each particle (velocityElementsCount * numParticles)
	unsigned int* colors; // 1D Array containing RGB data of each particle (colorElementsCount * numParticles)
	unsigned char* particleType; // 1D Array which denotes particle type (0 = Electron; 1 = Proton)

	

	//Particle Render Buffers
	GLuint VAO;
	GLuint positionBuffer;
	GLuint colorBuffer;

	Shader* shaderProgram;

	cudaGraphicsResource* positionResource;
	cudaGraphicsResource* colorResource;

	cudaEvent_t event;

	//Constants
	float coulomb_scalar;
	float yukawa_scalar;
	float yukawa_radius;
	float yukawa_cutoff;
	


};


#endif