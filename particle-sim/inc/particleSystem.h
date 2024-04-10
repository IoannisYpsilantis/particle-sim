#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

class ParticleSystem {
public:

	virtual ~ParticleSystem() {}

	virtual float* getPositions() = 0;

	virtual unsigned int* getColors() = 0;

	virtual void update(float timeDelta) = 0;

protected:
	int p_numParticles;

	float* positions;

	unsigned int* colors;

};


#endif