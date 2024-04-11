#include "particleSystemCpu.h"
#include <cstdlib>

ParticleSystemCPU::ParticleSystemCPU(int numParticles, int initMethod) {
	p_numParticles = numParticles;

	// Initialize Positions array
	int positionElementsCount = 4 * numParticles;
	positions = new float[positionElementsCount];
	//memset(positions, 0, positionElementsCount);

	// Initialize Colors array
	int colorElementsCount = 3 * numParticles;
	colors = new unsigned int[colorElementsCount];
	//memset(colors, 0, colorElementsCount);

	int velocityElementsCount = 3 * numParticles;
	velocities = new float[velocityElementsCount];

	// Initialize Particle Type array
	particleType = new unsigned char[numParticles];

	// Circular initialization
	if (initMethod == 0) {
		for (unsigned int i = 0; i < numParticles; i++) {
			float theta = (float)((numParticles - 1 - i) / (float)numParticles * 2.0 * 3.1415); // Ensure floating-point division
			positions[i * 4] = (float)cos(theta);
			positions[i * 4 + 1] = (float)sin(theta);
			positions[i * 4 + 2] = 1.0f;
			positions[i * 4 + 3] = 1.0f; // This will always stay as 1, it will be used for mapping 3D to 2D space

			colors[i * 3] = i % 255;
			colors[i * 3 + 1] = 255 - (i % 255);
			colors[i * 3 + 2] = 55;
		}

	}
	//Read from a file
	else if (initMethod == 1) {

	}
	// Random initialization in 3 dimensions
	else if (initMethod == 2) {
		for (unsigned int i = 0; i < numParticles; i++) {
			// Randomly initialize position in range [-1,1)
			positions[i * 4] = ((float)(rand() % 2000) - 1000.0) / 1000.0;
			positions[i * 4 + 1] = ((float)(rand() % 2000) - 1000.0) / 1000.0;
			positions[i * 4 + 2] = ((float)(rand() % 2000) - 1000.0) / 1000.0;
			positions[i * 4 + 3] = 1.0f; // This will always stay as 1, it will be used for mapping 3D to 2D space
			
			// Randomly initializes velocity in range [-0.0025,0.0025)
			velocities[i * 3] = ((float)(rand() % 500) - 250.0) / 100000.0;
			velocities[i * 3 + 1] = ((float)(rand() % 500) - 250.0) / 100000.0;
			velocities[i * 3 + 2] = ((float)(rand() % 500) - 250.0) / 100000.0;

			// Generates random number (either 0, 1) with 2/3 ratio being electrons
			particleType[i] = rand() % 3 % 2; 

			// Sets color based on particle type
			if (particleType[i]) { // If Proton
				colors[i * 3] = 255;
				colors[i * 3 + 1] = 0;
				colors[i * 3 + 2] = 0;
			}
			else { // Else electron
				colors[i * 3] = 0;
				colors[i * 3 + 1] = 180;
				colors[i * 3 + 2] = 255;
			}
		}
	}
	//Error bad method
	else {

	}

}
float* ParticleSystemCPU::getPositions(void) {
	return positions;
}

float* ParticleSystemCPU::getVelocities(void) {
	return velocities;
}

unsigned int* ParticleSystemCPU::getColors(void) {
	return colors;
}

void ParticleSystemCPU::update(float timeDelta) {
	for (int i = 0; i < p_numParticles; i++) {
		positions[i * 4] += velocities[i * 3];
		if (abs(positions[i * 4]) > 1) {
			velocities[i * 3] = -1 * velocities[i * 3];
		}
		positions[i * 4 + 1] += velocities[i * 3 + 1];
		if (abs(positions[i * 4 + 1]) > 1) {
			velocities[i * 3 + 1] = -1 * velocities[i * 3 + 1];
		}
		positions[i * 4 + 2] += velocities[i * 3 + 2];
		if (abs(positions[i * 4 + 2]) > 1) {
			velocities[i * 3 + 2] = -1 * velocities[i * 3 + 2];
		}
	}
}

ParticleSystemCPU::~ParticleSystemCPU() {
	p_numParticles = 0;
	delete[] positions;
	delete[] colors;
	delete[] velocities;
	delete[] particleType;
}