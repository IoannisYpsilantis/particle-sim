/*
The Buffer class is responsible for handling all buffers.
To this extent it houses both the VAO and VBO.
*/

#include "buffers.h"

Buffer::Buffer(float* positions, unsigned int* colors, int numParticles) {

	glGenVertexArrays(1, &ID);
	
	glBindVertexArray(ID);

	glGenBuffers(1, &positionBuffer);
	glGenBuffers(1, &colorBuffer);

	glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numParticles, positions, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(unsigned int) * 3 * numParticles, colors, GL_STATIC_DRAW);
	glVertexAttribIPointer(1, 3, GL_UNSIGNED_INT, 0, (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);


}

void Buffer::updatePositions(float* positions, int numParticles) {
	glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numParticles, positions, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Buffer::updateColors(unsigned int* colors, int numParticles) {
	glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(unsigned int) * 3 * numParticles, colors, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void Buffer::Delete() {
	glDeleteVertexArrays(1, &ID);
	glDeleteBuffers(1, &positionBuffer);
	glDeleteBuffers(1, &colorBuffer);
}