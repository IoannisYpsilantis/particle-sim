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
const bool render = true;
const int max_steps = -1; //Cutoff number of iterations, this is handy if rendering is false to determine a stop. Set to -1 to never terminate

// Physical parameters - mass
float proton_mass = 1.0f; //This is in atomic mass units 1amu ~ 1.67e-27 kg.
float electron_mass = 0.00055f;
float mass_ratio = 1836.15267343f; //This is the ratio with uncertainty of 6.0e-11.

// Physical parameters - forces
float residual_strong_range = 3; //In fentometers (10^-15 m)

// Window Parameters
const int width = 800;
const int height = 800;

int main(int argc, char** argv) {

    if (render == false && max_steps < 0) {
        printf("Warning: Can only exit program via ctrl-c. This is not recommended unless testing code.\n");
    }

    float* particles_pos;
    unsigned int* particles_col;
    ParticleSystem* system;
    if (useCPU) {
        system = new ParticleSystemCPU(numParticles, 2);
    }
    else {
        //Do GPU class initialization
    }

    GLFWwindow* window;
    Shader* shaderProgram;
    Buffer* buffers;

    if (render) {
        //Initialize GLFW 
        glfwInit();

        //This window is where we will view our graphics
        // (width, height, title, monitor, share)
        window = glfwCreateWindow(width, height, "Particle Simulation", NULL, NULL);

        //Check to make sure window was actually created, if not exit.
        if (!window) {
            glfwTerminate();
            delete system;
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



        particles_pos = system->getPositions();
        particles_col = system->getColors();

        shaderProgram = new Shader();

        buffers = new Buffer(particles_pos, particles_col, 1000);
    }
    

    

    int steps = 0;
    
    //This loop runs until the window is closed (or I guess if we make the program exit somehow)
    while (steps != max_steps && (!render || !glfwWindowShouldClose(window))) {
        system->update(0);
        

        if (render) {
            //glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            //Use the shader program
            shaderProgram->Activate();

            glDrawArrays(GL_POINTS, 0, 1000);

            //Swap the buffers so we see the new frame generated.
            glfwSwapBuffers(window);

            //We want to draw the points onto the screen:
            glfwPollEvents();

            buffers->updatePositions(particles_pos, 1000);
        }

        steps++;
        
    }

    //We have completed so we need to clean up.
    
    if (render) {
        //Delete the VBO and VAO
        buffers->Delete();
        delete buffers;

        //Delete the shader program
        shaderProgram->Delete();
        delete shaderProgram;

        //Terminate GLFW
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    

    //Delete the particleSystem
    delete system;
    
    return 0;

}