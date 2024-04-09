#include <stdio.h>
#include <math.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>

#include "shaderClass.h"




struct Particle {
    GLfloat x;
    GLfloat y;
    GLfloat z;
    GLuint r;
    GLuint g;
    GLuint b;
} typedef Particle;

int main(int argc, char** argv) {

    //Initialize particles
    Particle particles[1000];
    for (int i = 0; i < 1000; i++) {
        float theta = (float) (999 - i) / 1000 * 2 * 3.1415;
        particles[i].x = (GLfloat) cos(theta);
        particles[i].y = (GLfloat) sin(theta);
        particles[i].z = 1;
        particles[i].r = i % 255;
        particles[i].g = 255 - (i % 255);
        particles[i].b = 55;


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
    gladLoadGL();

    //What range of the screen we are actually drawing
    // (0, 0, 800, 600) is the full screen given the window size of 800x600
    glViewport(0, 0, 800, 600);



    Shader shaderProgram("../res/default.vert", "../res/default.frag");



    // VBO (Vertex Buffer Object) -> Buffer to send stuff to the GPU
    // VAO (Vertex Array Object) -> Stores pointers to VBOs and tells openGL how to interpret the data
    GLuint VAO, VBO;

    //One VBO so we put 1. 
    glGenVertexArrays(1, &VAO);

    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);


    glBufferData(GL_ARRAY_BUFFER, sizeof(particles), particles, GL_DYNAMIC_DRAW);


    //Tell OpenGL how to interpret the data
    // (index, size, type, normalized, stride, pointer)
    // index is the index of the vertex attribute we want to configure
    // size is the number of components per vertex attribute
    // type is the data type of each component in the array
    // normalized is whether the data should be normalized (are values between -1 and 1?)
    // stride is the byte offset between consecutive vertex attributes
    // pointer is the offset of the first component of the first vertex attribute in the array

    //I attempted to combine the color and position attributes here. It didn't work... Looking at the n-body example, 
    //It seems like different buffers entirely are used for color, and positions, thus, we should probably do the same. 
    //It will also be slightly different because the color should be either proton or electron. 
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(3*sizeof(GLfloat)));

    //Enable the vertex attribute
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

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
        shaderProgram.Activate();

        //Bind the VAO
        glBindVertexArray(VAO);

        glDrawArrays(GL_POINTS, 0, 1000);

        //Swap the buffers so we see the new frame generated.
        glfwSwapBuffers(window);

        




        //We want to draw the points onto the screen:
        
        glfwPollEvents();
        float temp_x = particles[0].x;
        float temp_y = particles[0].y;
        float temp_z = particles[0].z; 

        //for (int i = 0; i < 999; i++) {
        //    particles[i].x = particles[i + 1].x;
        //    particles[i].y = particles[i + 1].y;
        //    particles[i].z = particles[i + 1].z;
        //}

        //particles[999].x = temp_x;
        //particles[999].y = temp_y;
        //particles[999].z = temp_z;
        glBufferData(GL_ARRAY_BUFFER, sizeof(particles), particles, GL_DYNAMIC_DRAW);
    }

    //We have completed so we need to clean up.

    //Delete the VBO and VAO
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    //Delete the shader program
    shaderProgram.Delete();

    //Terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

}