#include <particleSystemCpu.h>

ParticleSystemCPU::ParticleSystemCPU(int numParticles, int initMethod, int seed) {
		p_numParticles = numParticles;

		// Initialize Positions array
		int positionElementsCount = 4 * numParticles;
		positions = new float[positionElementsCount];
#if doubleBuffer
		positions2 = new float[positionElementsCount];
#endif

		// Initialize Colors array
		int colorElementsCount = 3 * numParticles;
		colors = new unsigned int[colorElementsCount];

		int velocityElementsCount = 3 * numParticles;
		velocities = new float[velocityElementsCount];

		// Initialize Particle Type array
		particleType = new unsigned char[numParticles];


		// Circular initialization
		if (initMethod == 0) {
			for (unsigned int i = 0; i < numParticles; i++) {
				
				float theta = (float)((numParticles - 1 - i) / (float)numParticles * 2.0 * 3.1415); // Ensure floating-point division
				int pos_offset = 4;
				int col_offset = 3;

				positions[i * pos_offset] = (float)cos(theta) * boundingBox;
				positions[i * pos_offset + 1] = (float)sin(theta) * boundingBox;
				positions[i * pos_offset + 2] = 1.0f * boundingBox;
				positions[i * pos_offset + 3] = 1.0f * boundingBox; // This will always stay as 1, it will be used for mapping 3D to 2D space

				colors[i * col_offset] = i % 255;
				colors[i * col_offset + 1] = 255 - (i % 255);
				colors[i * col_offset + 2] = 55;
			}

		}
		//Hydrogen atoms
		else if (initMethod == 1) {
			if (seed != -1) {
				srand(seed);
			}
			int it = numParticles / 3;
			int pos_offset = 4;
			int vel_offset = 3;
			numProtons = it;
			numNeutrons = it;
			numElectrons = numParticles - 2 * it;
			for (unsigned int i = numElectrons; i < numElectrons+numProtons; i++) {
				
				//Pair up protons and neutrons
				float pos_X = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				float pos_Y = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				float pos_Z = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;

				positions[i * pos_offset] = pos_X;
				positions[(i + it) * pos_offset] = (float) (pos_X + yukawa_radius);

				positions[i * pos_offset + 1] = pos_Y;
				positions[(i + it) * pos_offset + 1] = (float)(pos_Y + yukawa_radius);

				positions[i * pos_offset + 2] = pos_Z;
				positions[(i + it) * pos_offset + 2] = (float)(pos_Z + yukawa_radius);
				
				particleType[i] = 1;
				particleType[i + it] = 2;
			}
			//Scatter in some electrons
			for (unsigned int i = 0; i < numElectrons; i++) {
				positions[i * pos_offset] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				positions[i * pos_offset + 1] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				positions[i * pos_offset + 2] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;

				particleType[i] = 0;
			}

			//Initialize velocities to 0 and give particales the proper color.
			for (unsigned int i = 0; i < numParticles; i++) {

				positions[i * pos_offset + 3] = 1.0f * boundingBox; // This will always stay as 1, it will be used for mapping 3D to 2D space

				velocities[i * vel_offset] = 0;
				velocities[i * vel_offset + 1] = 0;
				velocities[i * vel_offset + 2] = 0;

				// Sets color based on particle type
				if (particleType[i] == 0) { // If Electron
					colors[i * vel_offset] = ELECTRON_COLOR[0];
					colors[i * vel_offset + 1] = ELECTRON_COLOR[1];
					colors[i * vel_offset + 2] = ELECTRON_COLOR[2];
				}
				else if (particleType[i] == 1) { // If Proton
					colors[i * vel_offset] = PROTON_COLOR[0];
					colors[i * vel_offset + 1] = PROTON_COLOR[1];
					colors[i * vel_offset + 2] = PROTON_COLOR[2];
				}
				else {
					colors[i * vel_offset] = NEUTRON_COLOR[0]; //Else neutron
					colors[i * vel_offset + 1] = NEUTRON_COLOR[1];
					colors[i * vel_offset + 2] = NEUTRON_COLOR[2];
				}

			}
		}
		// Random initialization in 3 dimensions
		else if (initMethod == 2) {
			if (seed != -1) {
				srand(seed);
			}
			int it = numParticles / 3;
#if (orderedParticles)
			numProtons = it;
			numNeutrons = it;
			numElectrons = numParticles - 2 * it;
#else
			numProtons, numNeutrons, numElectrons = 0;
#endif
			for (unsigned int i = 0; i < numParticles; i++) {
				int pos_offset = 4;
				int vel_offset = 3;
				// Randomly initialize position in range [-1,1)
				positions[i * pos_offset] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				positions[i * pos_offset + 1] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				positions[i * pos_offset + 2] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				positions[i * pos_offset + 3] = 1.0f * boundingBox; // This will always stay as 1, it will be used for mapping 3D to 2D space

				// Randomly initializes velocity in range [-250000,250000)
				velocities[i * vel_offset] = ((float)(rand() % 500) - 250.0) * 1000.0;
				velocities[i * vel_offset + 1] = ((float)(rand() % 500) - 250.0) * 1000.0;
				velocities[i * vel_offset + 2] = ((float)(rand() % 500) - 250.0) * 1000.0;

				// Generates random number (either 0, 1, 2) from uniform dist
#if (orderedParticles)
				if (i < numElectrons) {
					particleType[i] = 0;
				}
				else if (i < numElectrons + numProtons) {
					particleType[i] = 1;
				}
				else {
					particleType[i] = 2;
				}
#else
				int type = rand() % 3;
				particleType[i] = type;
				if (type == 0) {
					numElectrons++;
				}
				else if (type == 1) {
					numProtons++;
				}
				else {
					numNeutrons++;
				}

#endif

				// Sets color based on particle type
				if (particleType[i] == 0) { // If Electron
					colors[i * vel_offset] = ELECTRON_COLOR[0];
					colors[i * vel_offset + 1] = ELECTRON_COLOR[1];
					colors[i * vel_offset + 2] = ELECTRON_COLOR[2];
				}
				else if (particleType[i] == 1) { // If Proton
					colors[i * vel_offset] = PROTON_COLOR[0];
					colors[i * vel_offset + 1] = PROTON_COLOR[1];
					colors[i * vel_offset + 2] = PROTON_COLOR[2];
				}
				else {
					colors[i * vel_offset] = NEUTRON_COLOR[0]; //Else neutron
					colors[i * vel_offset + 1] = NEUTRON_COLOR[1];
					colors[i * vel_offset + 2] = NEUTRON_COLOR[2];
				}
			}
		}

#if (doubleBuffer)
		for (int i = 0; i < p_numParticles * 4; i++) {
			positions2[i] = positions[i]; //This is really for the a (x,y,z,a) since it isn't overwritten
		}
		src = positions;
		dst = positions2;
#else
		src = positions;
		dst = positions;
#endif 

#if (RENDER_ENABLE)
		glGenVertexArrays(1, &VAO);

		glBindVertexArray(VAO);

		glGenBuffers(1, &positionBuffer);
		glGenBuffers(1, &colorBuffer);

		glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numParticles, positions, GL_STREAM_DRAW);

		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(unsigned int) * 3 * numParticles, colors, GL_STATIC_DRAW);


		glVertexAttribIPointer(1, 3, GL_UNSIGNED_INT, 0, (void*)0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		shaderProgram = new Shader();
#endif
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
#if (doubleBuffer)
			float dist_x = src[i * 4] - src[j * 4];
			float dist_y = src[i * 4 + 1] - src[j * 4 + 1];
			float dist_z = src[i * 4 + 2] - src[j * 4 + 2];
#else
			float dist_x = positions[i * 4] - positions[j * 4];
			float dist_y = positions[i * 4 + 1] - positions[j * 4 + 1];
			float dist_z = positions[i * 4 + 2] - positions[j * 4 + 2];
#endif

			float dist_square = square(dist_x) + square(dist_y) + square(dist_z);
			float dist = sqrt(dist_square);
			float force = 0.0f;
			if (i == j) {
				continue;
			}

			//Natural Coloumb force
			force += (float)coulomb_scalar / dist * charges[part_type] * charges[particleType[j]];

			//Strong Forces
			//P-N close attraction N-N close attraction 
			if (part_type != 0 && particleType[j] != 0) {
				if (dist < yukawa_cutoff) {
					force += yukawa_scalar * exp(-dist / yukawa_radius) / dist;
				}
				else {
					force -= yukawa_scalar * exp(-dist / yukawa_radius) / dist;
				}
				


			}

			force_x += force * dist_x / dist;
			force_y += force * dist_y / dist;
			force_z += force * dist_z / dist;


		}

		//Update velocities 
		velocities[i * 3] += force_x * inv_masses[part_type] * timeDelta;
		velocities[i * 3 + 1] += force_y * inv_masses[part_type] * timeDelta;
		velocities[i * 3 + 2] += force_z * inv_masses[part_type] * timeDelta;

		velocities[i * 3] *= dampingFactor;
		velocities[i * 3 + 1] *= dampingFactor;
		velocities[i * 3 + 2] *= dampingFactor;
	}
#if (doubleBuffer)
	for (int i = 0; i < p_numParticles; i++) {
		//Update positions from velocities only after all velocities were considered
		dst[i * 4] = src[i * 4] + velocities[i * 3] * timeDelta;
		if (abs(dst[i * 4]) > boundingBox) {
			velocities[i * 3] = -1 * velocities[i * 3];
		}
		dst[i * 4 + 1] = src[i * 4 + 1] + velocities[i * 3 + 1] * timeDelta;
		if (abs(dst[i * 4 + 1]) > boundingBox) {
			velocities[i * 3 + 1] = -1 * velocities[i * 3 + 1];
		}
		dst[i * 4 + 2] = src[i * 4 + 2] + velocities[i * 3 + 2] * timeDelta;
		if (abs(dst[i * 4 + 2]) > boundingBox) {
			velocities[i * 3 + 2] = -1 * velocities[i * 3 + 2];
		}
	}
#else
	for (int i = 0; i < p_numParticles; i++) {
		//Update positions from velocities only after all velocities were considered
		positions[i * 4] += velocities[i * 3] * timeDelta;
		if (abs(positions[i * 4]) > boundingBox) {
			velocities[i * 3] = -1 * velocities[i * 3];
		}
		positions[i * 4 + 1] += velocities[i * 3 + 1] * timeDelta;
		if (abs(positions[i * 4 + 1]) > boundingBox) {
			velocities[i * 3 + 1] = -1 * velocities[i * 3 + 1];
		}
		positions[i * 4 + 2] += velocities[i * 3 + 2] * timeDelta;
		if (abs(positions[i * 4 + 2]) > boundingBox) {
			velocities[i * 3 + 2] = -1 * velocities[i * 3 + 2];
		}
	}
#endif
}

void ParticleSystemCPU::flip() {
	float* temp = dst;
	dst = src;
	src = temp;
}

void ParticleSystemCPU::writecurpostofile(char* file, int steps, float milliseconds) {
	std::ofstream outfile(file);
#if doubleBuffer
	float* data = src;
#else
	float* data = positions;
#endif
	if (outfile.is_open()) {
		outfile << "particles:" << p_numParticles << " iterations:" << steps << " timing:" << milliseconds << " doubleBuffer:" << doubleBuffer << "\n";
		for (int i = 0; i < p_numParticles; i++) {
			outfile << data[i * 4] << " ";
			outfile << data[i * 4 + 1] << " ";
			outfile << data[i * 4 + 2] << " ";
			outfile << data[i * 4 + 3] << "\n";
		}
	}
	else {
		std::cerr << "Unable to open file: " << file << std::endl;
	}
}

void ParticleSystemCPU::display() {
	//If render wasn't specified this function does nothing. The code shouldn't allow it but it's a good check
#if (RENDER_ENABLE)
		//Update the positions
		glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
#if (doubleBuffer)
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * p_numParticles, dst, GL_STREAM_DRAW);
#else
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * p_numParticles, positions, GL_STREAM_DRAW);
#endif
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		shaderProgram->Activate();

		glPointSize(2.0);

		glDrawArrays(GL_POINTS, 0, p_numParticles);
#endif
}

ParticleSystemCPU::~ParticleSystemCPU() {
	p_numParticles = 0;
	delete[] positions;
	delete[] colors;
	delete[] velocities;
	delete[] particleType;

#if (RENDER_ENABLE)
		delete shaderProgram;
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &positionBuffer);
		glDeleteBuffers(1, &colorBuffer);
#endif
}