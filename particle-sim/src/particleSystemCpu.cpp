#include "particleSystemCpu.h"
#include <cstdlib>

ParticleSystemCPU::ParticleSystemCPU(int numParticles, int initMethod) {
	p_numParticles = numParticles;

	//Initialize Positions array
	int positionElementsCount = 4 * numParticles;
	positions = new float[positionElementsCount];
	//memset(positions, 0, positionElementsCount);

	//Initialize Colors array
	int colorElementsCount = 3 * numParticles;
	colors = new unsigned int[colorElementsCount];
	//memset(colors, 0, colorElementsCount);

	

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
	// Random initialization
	else if (initMethod == 2) {
		for (unsigned int i = 0; i < numParticles; i++) {
			positions[i * 4] = (float)(rand() % 2000 - 1000) / 1000.0;
			positions[i * 4 + 1] = (float)(rand() % 2000 - 1000) / 1000.0;
			positions[i * 4 + 2] = 1.0f;
			positions[i * 4 + 3] = 1.0f; // This will always stay as 1, it will be used for mapping 3D to 2D space

			colors[i * 3] = i % 255;
			colors[i * 3 + 1] = 255 - (i % 255);
			colors[i * 3 + 2] = 55;
		}
	}
	//Error bad method
	else {

	}

}
float* ParticleSystemCPU::getPositions() {
	return positions;
}

unsigned int* ParticleSystemCPU::getColors() {
	return colors;
}

void ParticleSystemCPU::update(float timeDelta) {
	float temp_x = positions[0];
	float temp_y = positions[1];
	float temp_z = positions[2];
	float temp_a = positions[3];

	for (int i = 0; i < (p_numParticles - 1); i++) {
		positions[i * 4] = positions[i * 4 + 4];
		positions[i * 4 + 1] = positions[i * 4 + 5];
		positions[i * 4 + 2] = positions[i * 4 + 6];
		positions[i * 4 + 3] = positions[i * 4 + 7];
	}

	positions[(p_numParticles - 1) * 4] = temp_x;
	positions[(p_numParticles - 1) * 4 + 1] = temp_y;
	positions[(p_numParticles - 1) * 4 + 2] = temp_z;
	positions[(p_numParticles - 1) * 4 + 3] = temp_a;
}

ParticleSystemCPU::~ParticleSystemCPU() {
	p_numParticles = 0;
	delete[] positions;
	delete[] colors;
}