#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>

#include "shaderClass.h"
#include "buffers.h"
#include "particleSystem.h"
#include "particleSystemCpu.h"





int main(int argc, char** argv) {

    int num_particles = 1000;

    bool cpu = true;

   

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

    float* particles_pos;
    unsigned int* particles_col;
    ParticleSystem* system;
    if (cpu) {
        system = new ParticleSystemCPU(1000, 0);
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

    //This loop runs until the window is closed (or I guess if we make the program exit somehow)
    while(!glfwWindowShouldClose(window)) {
        glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
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