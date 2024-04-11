#ifndef PARTICLESYSTEMCPU_H
#define PARTICLESYSTEMCPU_H

#include <cstring>
#include <math.h>

#include "particleSystem.h"


class ParticleSystemCPU : public ParticleSystem
{
public:
	ParticleSystemCPU(int numParticles, int init_method);

	~ParticleSystemCPU(void);

	float* getPositions(void);

	float* getVelocities(void);

	unsigned int* getColors(void);

	void update(float timeDelta);


protected:
	int p_numParticles;

	// Particle Data
	float* positions; // 1D Array containing spacial data of each particle (positionElementsCount * numParticles)
	float* velocities; // 1D Array containing velocity data of each particle (velocityElementsCount * numParticles)
	unsigned int* colors; // 1D Array containing RGB data of each particle (colorElementsCount * numParticles)
	unsigned char* particleType; // 1D Array which denotes particle type (0 = Electron; 1 = Proton)
};

#endif