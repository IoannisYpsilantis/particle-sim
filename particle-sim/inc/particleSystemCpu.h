#ifndef PARTICLESYSTEMCPU_H
#define PARTICLESYSTEMCPU_H

#include <cstring>
#include <math.h>

#include "particleSystem.h"


class ParticleSystemCPU : public ParticleSystem
{
public:
	ParticleSystemCPU(int numParticles, int init_method);

	~ParticleSystemCPU();

	float* getPositions();

	unsigned int* getColors();

	void update(float timeDelta);


protected:
	int p_numParticles;

	float* positions;

	unsigned int* colors;
};





#endif