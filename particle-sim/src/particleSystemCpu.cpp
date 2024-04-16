#include "particleSystemCpu.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>

ParticleSystemCPU::ParticleSystemCPU(int numParticles, int initMethod, int seed) {
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
	
	//refer to equations.ipynb to see why this value is what it is.
	coulomb_scalar = 2.310272969e-4; //N*picometers^2

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
		if (seed != -1) {
			srand(seed);
		}
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

			// Generates random number (either 0, 1, 2) from uniform dist
			particleType[i] = rand() % 3 % 2; 

			// Sets color based on particle type
			if (particleType[i] == 0) { // If Electron
				colors[i * 3] = 0;
				colors[i * 3 + 1] = 180;
				colors[i * 3 + 2] = 255;
				
			}
			else if (particleType[i] == 1) { // If Proton
				colors[i * 3] = 255;
				colors[i * 3 + 1] = 0;
				colors[i * 3 + 2] = 0;
			}
			else {
				colors[i * 3] = 80; //Else neutron
				colors[i * 3 + 1] = 80;
				colors[i * 3 + 2] = 80;

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

float square(float val) {
	return pow(val, 2);
}

void ParticleSystemCPU::update(float timeDelta) {
	for (int i = 0; i < p_numParticles; i++) {
		//Update velocities
		int part_type = particleType[i];
		float force_x = 0.0f;
		float force_y = 0.0f;
		float force_z = 0.0f;
		for (int j = 0; j < p_numParticles; j++) {
			if (i == j) {
				continue;
			}
			//float dist_square = square(positions[i] - positions[j]) + square(positions[i + 1] - positions[j + 1]) + square(positions[i + 2] - positions[j + 2]);
			float dist_square = square(positions[i] - positions[j]) + square(positions[i + 1] - positions[j + 1]);
			float dist = sqrt(dist_square);
			//Natural Coloumb forces
			if ((part_type == 0 || part_type == 1) && (particleType[j] == 0 || particleType[j] == 1)) {
				float force = 0;
				if (particleType[i] == particleType[j]) {
					force = coulomb_scalar / dist_square;
				}
				else {
					force = -coulomb_scalar / dist_square;
				}
				//Calculate net forces.
				float dist_x = positions[i] - positions[j];
				float dist_y = positions[i + 1] - positions[j + 1];
				force_x += force * dist_x / dist;
				force_y += force * dist_y / dist;
				
			}

		}
		//Update velocities 
		if (particleType[i] == 0) {
			velocities[i] += force_x * 1.09776e30 * 1e-12 * timeDelta;
			velocities[i + 1] += force_y * 1.09776e30 * 1e-12 * timeDelta;
			velocities[i + 2] += force_z * 1.09776e30 * 1e-12 * timeDelta;
		}
		else {
			velocities[i] += force_x * 5.978638e26 * 1e-12 *timeDelta;
			velocities[i + 1] += force_y * 5.978638e26 * 1e-12 * timeDelta;
			velocities[i + 2] += force_z * 5.978638e26 * 1e-12 * timeDelta;
		}


		//Update positions from velocities
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

void ParticleSystemCPU::writecurpostofile(char* file) {
	std::ofstream outfile(file);

	if (outfile.is_open()) {
		for (int i = 0; i < p_numParticles; i++) {
			outfile << positions[i * 4] << " ";
			outfile << positions[i * 4 + 1] << " ";
			outfile << positions[i * 4 + 2] << " ";
			outfile << positions[i * 4 + 3] << "\n";
		}
	}
	else {
		std::cerr << "Unable to open file: " << file << std::endl;
	}
}

ParticleSystemCPU::~ParticleSystemCPU() {
	p_numParticles = 0;
	delete[] positions;
	delete[] colors;
	delete[] velocities;
	delete[] particleType;
}