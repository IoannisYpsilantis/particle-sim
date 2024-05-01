#ifndef ORDERED_H
#define ORDERED_H

#include "common.h"
#include "particleSystemGpu.h"

#include <cmath>


// GPU Library Includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
/*
__global__ void update_electrons(float timeDelta, int numParticles, int numE, int numP, float* positions, float* velocities, unsigned char* particleType);

__global__ void update_protons(float timeDelta, int numParticles, int numE, int numP, int numN, float* positions, float* velocities, unsigned char* particleType);

__global__ void update_neutrons(float timeDelta, int numParticles, int numE, int numP, int numN, float* positions, float* velocities, unsigned char* particleType);

*/
#endif