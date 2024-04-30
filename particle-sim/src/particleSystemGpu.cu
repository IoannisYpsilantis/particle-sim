#include "particleSystemGpu.h"

__constant__ float d_inv_masses[3];
__constant__ float d_charges[3];

__global__ void update_naive(float timeDelta, int numParticles, float* positions, float* velocities, unsigned char* particleType) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < numParticles) {
		int part_type = particleType[gid];
		float force_x = 0.0;
		float force_y = 0.0;
		float force_z = 0.0;
		for (int j = 0; j < numParticles; j++) {
			float dist_x = positions[gid * 4] - positions[j * 4];
			float dist_y = positions[gid * 4 + 1] - positions[j * 4 + 1];
			float dist_z = positions[gid * 4 + 2] - positions[j * 4 + 2];
			float dist_square = (dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z);
			float dist = sqrt(dist_square);
			if (gid == j) {
				continue;
			}
			float force = 0.0;
			//Coulomb force
			force += (float)coulomb_scalar / dist * d_charges[part_type] * d_charges[particleType[j]];



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
			//Break force into components
			force_x += force * dist_x / dist;
			force_y += force * dist_y / dist;
			force_z += force * dist_z / dist;
		}

		//Update velocities
		velocities[gid * 3] += force_x * d_inv_masses[part_type] * timeDelta;
		velocities[gid * 3 + 1] += force_y * d_inv_masses[part_type] * timeDelta;
		velocities[gid * 3 + 2] += force_z * d_inv_masses[part_type] * timeDelta;

		velocities[gid * 3] *= dampingFactor;
		velocities[gid * 3 + 1] *= dampingFactor;
		velocities[gid * 3 + 2] *= dampingFactor;
	}

}

__global__ void update_positions(float timeDelta, float * positions, float *velocities) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	//Update positions from velocities
	positions[gid * 4] += velocities[gid * 3] * timeDelta;
	if (abs(positions[gid * 4]) > boundingBox) {
		velocities[gid * 3] = -1 * velocities[gid * 3];
	}
		
	positions[gid * 4 + 1] += velocities[gid * 3 + 1] * timeDelta;
	if (abs(positions[gid * 4 + 1]) > boundingBox) {
		velocities[gid * 3 + 1] = -1 * velocities[gid * 3 + 1];
	}

	positions[gid * 4 + 2] += velocities[gid * 3 + 2] * timeDelta;
	if (abs(positions[gid * 4 + 2]) > boundingBox) {
		velocities[gid * 3 + 2] = -1 * velocities[gid * 3 + 2];
	}
}




ParticleSystemGPU::ParticleSystemGPU(int numParticles, int initMethod, int seed) {
		p_numParticles = numParticles;

		blockSize = TILE_SIZE;
		gridSize = (int)ceil((float)numParticles / (float)TILE_SIZE);
		cudaEventCreate(&event);

		// Initialize Positions array
		int positionElementsCount = 4 * numParticles;
		positions = new float[positionElementsCount];

		// Initialize Colors array
		int colorElementsCount = 3 * numParticles;
		colors = new unsigned int[colorElementsCount];

		int velocityElementsCount = 3 * numParticles;
		velocities = new float[velocityElementsCount];

		// Initialize Particle Type array
		particleType = new unsigned char[numParticles];

		// Circular initialization
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
			for (unsigned int i = 0; i < it; i++) {

				//Pair up protons and neutrons
				float pos_X = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				float pos_Y = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				float pos_Z = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;

				positions[i * pos_offset] = pos_X;
				positions[(i + it) * pos_offset] = (float)(pos_X + yukawa_radius);

				positions[i * pos_offset + 1] = pos_Y;
				positions[(i + it) * pos_offset + 1] = (float)(pos_Y + yukawa_radius);

				positions[i * pos_offset + 2] = pos_Z;
				positions[(i + it) * pos_offset + 2] = (float)(pos_Z + yukawa_radius);

				particleType[i] = 1;
				particleType[i + it] = 2;
			}
			//Scatter in some electrons
			for (unsigned int i = 2 * it - 1; i < numParticles; i++) {
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
			for (unsigned int i = 0; i < numParticles; i++) {
				int pos_offset = 4;
				int vel_offset = 3;
				// Randomly initialize position in range [-1,1)
				positions[i * pos_offset] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				positions[i * pos_offset + 1] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				positions[i * pos_offset + 2 ] = ((float)(rand() % 2000) - 1000.0) / 1000.0 * boundingBox;
				positions[i * pos_offset + 3 ] = 1.0f * boundingBox; // This will always stay as 1, it will be used for mapping 3D to 2D space

				// Randomly initializes velocity in range [-250000,250000)
				velocities[i * vel_offset] = ((float)(rand() % 500) - 250.0) * 1000.0;
				velocities[i * vel_offset + 1] = ((float)(rand() % 500) - 250.0) * 1000.0;
				velocities[i * vel_offset + 2] = ((float)(rand() % 500) - 250.0) * 1000.0;

				// Generates random number (either 0, 1, 2) from uniform dist
				particleType[i] = rand() % 3;
				//particleType[i] = 2;

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
		//Error bad method
		else {
			std::cerr << "Bad Initialization";
			}

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

		//Initialize device

		cudaMemcpyToSymbol(d_inv_masses, inv_masses, 3 * sizeof(float));
		cudaMemcpyToSymbol(d_charges, charges, 3 * sizeof(float));

#if (RENDER_ENABLE)
		cudaGraphicsGLRegisterBuffer(&positionResource, positionBuffer, cudaGraphicsMapFlagsNone);
#else
		cudaMalloc(&d_positions, positionElementsCount * sizeof(float));
		cudaMemcpy(d_positions, positions, positionElementsCount * sizeof(float), cudaMemcpyHostToDevice);
#endif

		cudaMalloc(&d_velocities, velocityElementsCount * sizeof(float));
		cudaMemcpy(d_velocities, velocities, velocityElementsCount * sizeof(float), cudaMemcpyHostToDevice);

#if (RENDER_ENABLE)
		cudaGraphicsGLRegisterBuffer(&colorResource, colorBuffer, cudaGraphicsMapFlagsNone);
#else
		cudaMalloc(&d_colors, colorElementsCount * sizeof(unsigned int));
		cudaMemcpy(d_colors, colors, colorElementsCount * sizeof(unsigned int), cudaMemcpyHostToDevice);
#endif
	
		cudaMalloc(&d_particleType, numParticles * sizeof(unsigned char));
		cudaMemcpy(d_particleType, particleType, numParticles * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

float* ParticleSystemGPU::getPositions() {
#if (RENDER_ENABLE)
		size_t Size;
		cudaGraphicsMapResources(1, &positionResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &Size, positionResource);
#endif
	
		int numBytes = p_numParticles * 4 * sizeof(float);
		cudaMemcpy(positions, d_positions, numBytes, cudaMemcpyDeviceToHost);

#if (RENDER_ENABLE)
		cudaGraphicsUnmapResources(1, &positionResource, 0);
#endif
		return positions;
}

float* ParticleSystemGPU::getVelocities() {
		int numBytes = p_numParticles * 3 * sizeof(float);
		cudaMemcpy(velocities, d_velocities, numBytes, cudaMemcpyDeviceToHost);
		return velocities;
}

unsigned int* ParticleSystemGPU::getColors() {
#if (RENDER_ENABLE)
		size_t Size;
		cudaGraphicsMapResources(1, &colorResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &Size, colorResource);
#endif
	
		int numBytes = p_numParticles * 3 * sizeof(unsigned int);
		cudaMemcpy(colors, d_colors, numBytes, cudaMemcpyDeviceToHost);

#if (RENDER_ENABLE)
		cudaGraphicsUnmapResources(1, &colorResource, 0);
#endif
	return colors;
}



void ParticleSystemGPU::update(float timeDelta) {
#if (RENDER_ENABLE)
		size_t Size;
		cudaGraphicsMapResources(1, &positionResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &Size, positionResource);
#endif
		update_naive<<<gridSize, blockSize>>>(timeDelta, p_numParticles, d_positions, d_velocities, d_particleType);

		update_positions<<<gridSize, blockSize>>>(timeDelta, d_positions, d_velocities);
		//std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

		cudaError_t cudaStatusFlag = cudaGetLastError();
		if (cudaStatusFlag != cudaSuccess) {
			std::cerr << "Kernel failed: " << cudaGetErrorString(cudaStatusFlag) << std::endl;

			// Do cudaFree here

			//return false;

			// Should probably make this function return a boolean which indicates success
			// Stop sim if error encountered
		}

		cudaEventRecord(event);

		cudaEventSynchronize(event);

#if (RENDER_ENABLE)
		cudaGraphicsUnmapResources(1, &positionResource, 0);
#endif
}

void ParticleSystemGPU::flip() {
	//To do
}



void ParticleSystemGPU::writecurpostofile(char* file, int steps, float milliseconds) {
		getPositions();
		std::ofstream outfile(file);

		if (outfile.is_open()) {
			outfile << "particles:" << p_numParticles << " iterations:" << steps << " timing:" << milliseconds << "\n";
			for (int i = 0; i < p_numParticles; i++) {
#if (STORAGE_TYPE && !RENDER_ENABLE)
				outfile << positions[i] << " ";
				outfile << positions[i + p_numParticles] << " ";
				outfile << positions[i + 2 * p_numParticles] << " ";
				outfile << positions[i + 3 * p_numParticles] << "\n";
#else
				outfile << positions[i * 4] << " ";
				outfile << positions[i * 4 + 1] << " ";
				outfile << positions[i * 4 + 2] << " ";
				outfile << positions[i * 4 + 3] << "\n";
#endif
			}
		}
		else {
			std::cerr << "Unable to open file: " << file << std::endl;
		}
}

	
void ParticleSystemGPU::display() {
#if (RENDER_ENABLE)
		//Positions are already updated since we work directly on the data!

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		shaderProgram->Activate();

		glPointSize(2.0);
		
		glDrawArrays(GL_POINTS, 0, p_numParticles);
#endif
}

ParticleSystemGPU::~ParticleSystemGPU() {
	p_numParticles = 0;
	delete[] positions;
	delete[] colors;
	delete[] velocities;
	delete[] particleType;

	//VBO will handle positions and colors buffers if we rendered.
	cudaFree(d_velocities);
	cudaFree(d_particleType);

	cudaEventDestroy(event);
#if (RENDER_ENABLE)
		delete shaderProgram;
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &positionBuffer);
		glDeleteBuffers(1, &colorBuffer);
#else
		cudaFree(d_positions);
		cudaFree(d_colors);
#endif
	
}