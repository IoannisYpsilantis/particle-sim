#ifndef PARTICLESYSTEMGPU_H
#define PARTICLESYSTEMGPU_H

#include "particleSystem.h"

#include "buffers.h"

class ParticleSystemGPU : public ParticleSystem
{
public:
	ParticleSystemGPU(int numParticles, int init_method, int seed, bool render);

	~ParticleSystemGPU(void);

	void assignBuffer(Buffer* buffer);

	float* getPositions(void);

	float* getVelocities(void);

	unsigned int* getColors(void);

	void update(float timeDelta);

	void writecurpostofile(char* file);

protected:
	int p_numParticles;
	float coulomb_scalar;
	float yukawa_scalar;
	float yukawa_radius;
	float yukawa_cutoff;

	//Kernel specs
	int blockSize;
	int gridSize;

	//Particle Data Host
	float* positions; // 1D Array containing spacial data of each particle (positionElementsCount * numParticles)
	float* velocities; // 1D Array containing velocity data of each particle (velocityElementsCount * numParticles)
	unsigned int* colors; // 1D Array containing RGB data of each particle (colorElementsCount * numParticles)
	unsigned char* particleType; // 1D Array which denotes particle type (0 = Electron; 1 = Proton)

	//Particle Data Device
	float* d_positions;
	float* d_velocities;
	unsigned int* d_colors;
	unsigned char* d_particleType;

	Buffer* p_buffer;
	bool p_render;


};


#endif