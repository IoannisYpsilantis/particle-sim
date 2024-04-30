#ifndef PARTICLESYSTEMCPU_H
#define PARTICLESYSTEMCPU_H

// Standard Library Includes
#include <cstring>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>

// Visualization Includes
#include <glad/glad.h>

//Program Includes
#include <common.h>
#include <buffers.h>
#include <particleSystem.h>

class ParticleSystemCPU : public ParticleSystem
{
public:
		ParticleSystemCPU(int numParticles, int init_method, int seed);

		~ParticleSystemCPU(void);// Possibly don't need to redeclare this here cause its in particleSystem.h

		float* getPositions(void);// Possibly don't need to redeclare this here cause its in particleSystem.h

		float* getVelocities(void);

		unsigned int* getColors(void);// Possibly don't need to redeclare this here cause its in particleSystem.h

		void update(float timeDelta);// Possibly don't need to redeclare this here cause its in particleSystem.h

		void flip();

		void writecurpostofile(char* file, int steps, float milliseconds);// Possibly don't need to redeclare this here cause its in particleSystem.h

		void display(); // Possibly don't need to redeclare this here cause its in particleSystem.h

protected:

};

#endif