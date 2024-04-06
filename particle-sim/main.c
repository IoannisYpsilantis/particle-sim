#include <stdio.h>

#include <glad/gl.h>
#include <GLFW/glfw3.h>





// Vertex Shader source code
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";
//Fragment Shader source code
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(0.8f, 0.3f, 0.02f, 1.0f);\n"
"}\n\0";



struct Particle {
    GLfloat x;
    GLfloat y;
    GLfloat z;
} typedef Particle;

int main(int argc, char** argv) {

    //Initialize particles
    Particle particles[1000];
    for (int i = 0; i < 1000; i++) {
        particles[i].x = -1.0f + i % 10 * 0.2f;
        particles[i].y = -1.0f + ((i / 10) % 10) * 0.2f;
        particles[i].z = -1.0f + i / 100 * 0.2f;
    }
    //Initialize GLFW 
    glfwInit();

    //This window is where we will view our graphics
    // (width, height, title, monitor, share)
    GLFWwindow* window = glfwCreateWindow(800, 600, "Hello World", NULL, NULL);

    //Check to make sure window was actually created, if not exit.
    if (!window) {
        glfwTerminate();
        return -1;
    }

    //Make the window the context for OpenGL
    glfwMakeContextCurrent(window);

    //Load OpenGL functions
    gladLoadGL(glfwGetProcAddress);

    //What range of the screen we are actually drawing
    // (0, 0, 800, 600) is the full screen given the window size of 800x600
    glViewport(0, 0, 800, 600);



    // Create Vertex Shader Object and get its reference
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// Attach Vertex Shader source to the Vertex Shader Object
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	// Compile the Vertex Shader into machine code
	glCompileShader(vertexShader);

	// Create Fragment Shader Object and get its reference
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	// Attach Fragment Shader source to the Fragment Shader Object
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	// Compile the Vertex Shader into machine code
	glCompileShader(fragmentShader);

	// Create Shader Program Object and get its reference
	GLuint shaderProgram = glCreateProgram();
	// Attach the Vertex and Fragment Shaders to the Shader Program
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	// Wrap-up/Link all the shaders together into the Shader Program
	glLinkProgram(shaderProgram);

	// Delete the now useless Vertex and Fragment Shader objects
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

    // VBO (Vertex Buffer Object) -> Buffer to send stuff to the GPU
    // VAO (Vertex Array Object) -> Stores pointers to VBOs and tells openGL how to interpret the data
    GLuint VAO, VBO;

    //One VBO so we put 1. 
    glGenVertexArrays(1, &VAO);

    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(particles), 0, GL_STATIC_DRAW);


    //Tell OpenGL how to interpret the data
    // (index, size, type, normalized, stride, pointer)
    // index is the index of the vertex attribute we want to configure
    // size is the number of components per vertex attribute
    // type is the data type of each component in the array
    // normalized is whether the data should be normalized (are values between -1 and 1?)
    // stride is the byte offset between consecutive vertex attributes
    // pointer is the offset of the first component of the first vertex attribute in the array
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);

    //Enable the vertex attribute
    glEnableVertexAttribArray(0);

    //cudaGLRegisterBufferObject(VBO);

    //float* particles_dev;

    //cudaGLMapBufferObject((void**)&particles_dev, VBO);


    

   

    //Specify the color of the window
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    //Clear the window to the color we set in glClearColor
    glClear(GL_COLOR_BUFFER_BIT);

    //Swap the buffers so we see the new frame generated.
    glfwSwapBuffers(window);

    //This loop runs until the window is closed (or I guess if we make the program exit somehow)
    while(!glfwWindowShouldClose(window)) {
        glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //Use the shader program
        glUseProgram(shaderProgram);

        //Bind the VAO
        glBindVertexArray(VAO);

        glDrawArrays(GL_POINTS, 0, 1000);

        //Swap the buffers so we see the new frame generated.
        glfwSwapBuffers(window);





        //We want to draw the points onto the screen:
        
        glfwPollEvents();
    }

    //We have completed so we need to clean up.

    //Delete the VBO and VAO
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    //Delete the shader program
    glDeleteProgram(shaderProgram);

    //Terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

}