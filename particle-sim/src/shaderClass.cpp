/*
The shader class is responsible for initializing and activating the shaders.
Currently it takes in the position of the particle along with its color from two different buffers and puts it 2 dimensionally on the screen.
*/

#include <shaderClass.h>

const char vertexShaderSource[] = { "#version 330 core\n"
"layout (location = 0) in vec4 aPos;\n"
"layout (location = 1) in uvec3 color;\n"
"out vec3 vertexColor;\n"
"void main()\n"
"{\n"
"   gl_Position = aPos / 1000.0;\n" //THE DIVISOR HERE SHOULD MATCH THE BOUNDINGBOX VARIABLE IN COMMON.H
"   vertexColor = color / 255.0;\n"
"}" };


const char fragmentShaderSource[] = { "#version 330 core\n"
"in vec3 vertexColor;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(vertexColor, 1.0f);\n"
"}\n" };

// Constructor that build the Shader Program from 2 different shaders
Shader::Shader()
{
	// Read vertexFile and fragmentFile and store the strings
	const char* vertexCode = vertexShaderSource;
	const char* fragmentCode = fragmentShaderSource;

	// Convert the shader source strings into character arrays
	//const char* vertexSource = vertexCode.c_str();
	//const char* fragmentSource = fragmentCode.c_str();

	// Create Vertex Shader Object and get its reference
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// Attach Vertex Shader source to the Vertex Shader Object
	glShaderSource(vertexShader, 1, &vertexCode, NULL);
	// Compile the Vertex Shader into machine code
	glCompileShader(vertexShader);

	// Create Fragment Shader Object and get its reference
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	// Attach Fragment Shader source to the Fragment Shader Object
	glShaderSource(fragmentShader, 1, &fragmentCode, NULL);
	// Compile the Vertex Shader into machine code
	glCompileShader(fragmentShader);

	// Create Shader Program Object and get its reference
	ID = glCreateProgram();
	// Attach the Vertex and Fragment Shaders to the Shader Program
	glAttachShader(ID, vertexShader);
	glAttachShader(ID, fragmentShader);
	// Wrap-up/Link all the shaders together into the Shader Program
	glLinkProgram(ID);

	// Delete the now useless Vertex and Fragment Shader objects
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

}

// Activates the Shader Program
void Shader::Activate()
{
	glUseProgram(ID);
}

// Deletes the Shader Program
void Shader::Delete()
{
	glDeleteProgram(ID);
}
