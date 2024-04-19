/*
The shader class is responsible for initializing and activating the shaders. 
Currently it takes in the position of the particle along with its color from two different buffers and puts it 2 dimensionally on the screen. 
*/


#ifndef __SHADER_CLASS_H
#define __SHADER_CLASS_H

#include <glad/glad.h>
#include <cerrno>

class Shader
{
public:
	// Reference ID of the Shader Program
	GLuint ID;

	// Constructor that build the Shader Program from 2 different shaders
	Shader();

	// Activates the Shader Program
	void Activate();

	// Deletes the Shader Program
	void Delete();
};
#endif
