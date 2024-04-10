/*
The Buffer class is responsible for handling all buffers.
To this extent it houses both the VAO and VBO.
*/

#ifndef BUFFER_CLASS_H
#define BUFFER_CLASS_H

#include<glad/glad.h>


class Buffer
{
public:
	// Reference ID to the Buffers
	GLuint ID, positionBuffer, colorBuffer;

	Buffer(float* positions, unsigned int* colors, int numParticles);

	void updatePositions(float* positions, int numParticles);

	void updateColors(unsigned int* colors, int numParticles);

	void Delete();

};












#endif