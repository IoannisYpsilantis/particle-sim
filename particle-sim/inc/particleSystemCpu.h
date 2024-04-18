#ifndef PARTICLESYSTEMCPU_H
#define PARTICLESYSTEMCPU_H

#include <cstring>
#include <math.h>
#include<glad/glad.h>

#include "particleSystem.h"



class ParticleSystemCPU : public ParticleSystem
{
public:
	ParticleSystemCPU(int numParticles, int init_method, int seed, bool render);

	~ParticleSystemCPU(void);

	float* getPositions(void);

	float* getVelocities(void);

	unsigned int* getColors(void);

	void update(float timeDelta);

	void writecurpostofile(char* file);

	void display();

	

protected:
	int p_numParticles;
	bool p_render;
	

	// Particle Data
	float* positions; // 1D Array containing spacial data of each particle (positionElementsCount * numParticles)
	float* velocities; // 1D Array containing velocity data of each particle (velocityElementsCount * numParticles)
	unsigned int* colors; // 1D Array containing RGB data of each particle (colorElementsCount * numParticles)
	unsigned char* particleType; // 1D Array which denotes particle type (0 = Electron; 1 = Proton)

	// Rendering buffers
	GLuint VAO;
	GLuint positionBuffer;
	GLuint colorBuffer;

	Shader* shaderProgram;

	

	//Constants
	float coulomb_scalar;
	float yukawa_scalar;
	float yukawa_radius;
	float yukawa_cutoff;
};

#endif