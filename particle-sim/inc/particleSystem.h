#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include "shaderClass.h"
#include<glad/glad.h>

class ParticleSystem {
public:

	virtual ~ParticleSystem() {}

	virtual float* getPositions() = 0;

	virtual unsigned int* getColors() = 0;

	virtual void update(float timeDelta) = 0;

	virtual void writecurpostofile(char* file) = 0;
	
	virtual void display() = 0;

	

protected:
	
	int p_numParticles;
	bool p_render;

	//Necessary buffers
	float* positions;

	unsigned int* colors;

	//Shader buffers
	GLuint VAO;
	GLuint positionBuffer;
	GLuint colorBuffer;

	Shader* shaderProgram;

	//Necessary constants
	float coulomb_scaler;
	float yukawa_scalar;
	float yukawa_radius;
	float yukawa_cutoff;

};


#endif