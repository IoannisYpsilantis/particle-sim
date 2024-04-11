// Includes
#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>

#include "shaderClass.h"
#include "buffers.h"
#include "particleSystem.h"
#include "particleSystemCpu.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// Environment Parameters
const int numParticles = 1000;
const bool useCPU = true;

// Window Parameters
const int width = 800;
const int height = 800;

int main(int argc, char** argv) {

    //Initialize GLFW 
    glfwInit();

    //This window is where we will view our graphics
    // (width, height, title, monitor, share)
    GLFWwindow* window = glfwCreateWindow(width, height, "Particle Simulation", NULL, NULL);

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
    glViewport(0, 0, width, height);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    float* particles_pos;
    unsigned int* particles_col;
    ParticleSystem* system;
    if (useCPU) {
        system = new ParticleSystemCPU(numParticles, 2);
    }
    else {
        //Do GPU class initialization
    }

    particles_pos = system->getPositions();
    particles_col = system->getColors();

    Shader shaderProgram;

    Buffer buffers(particles_pos, particles_col, 1000);

    //Specify the color of the window
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    //Clear the window to the color we set in glClearColor
    glClear(GL_COLOR_BUFFER_BIT);

    //Swap the buffers so we see the new frame generated.
    glfwSwapBuffers(window);
    GLint m_viewport[4];

    
    //This loop runs until the window is closed (or I guess if we make the program exit somehow)
    while(!glfwWindowShouldClose(window)) {
        //glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //Use the shader program
        shaderProgram.Activate();

        glDrawArrays(GL_POINTS, 0, 1000);

        //Swap the buffers so we see the new frame generated.
        glfwSwapBuffers(window);

        //We want to draw the points onto the screen:
        glfwPollEvents();
        
        system->update(0);

        buffers.updatePositions(particles_pos, 1000);
    }

    //We have completed so we need to clean up.
    //Delete the VBO and VAO
    buffers.Delete();

    //Delete the shader program
    shaderProgram.Delete();

    //Delete the particleSystem
    delete system;

    //Terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

}