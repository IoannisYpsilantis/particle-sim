// Includes
#include <stdio.h>
#include <ctime>
#include <iostream>



#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "shaderClass.h"
#include "buffers.h"
#include "particleSystem.h"
#include "particleSystemCpu.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// Environment Parameters
const int numParticles = 5000;
const bool useCPU = true;
const bool render = true;
const bool saveFinal = false; //Save final positions to a designated folder
const int max_steps = -1; //Cutoff number of iterations, this is handy if rendering is false to determine a stop. Set to -1 to never terminate
const int seed = 42; //Seed for run, set to 1 for random generation.

     

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
        system = new ParticleSystemCPU(numParticles, 2, seed);
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
    //Set up timers
    clock_t cpu_start, cpu_end;
    cudaEvent_t gpu_start, gpu_end;
    float milliseconds;
    if (useCPU) {
        cpu_start = clock();
    }
    else {
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_end);
        cudaEventRecord(gpu_start);
    }
    
    //This loop runs until the window is closed (or I guess if we make the program exit somehow)
    while (steps != max_steps && (!render || !glfwWindowShouldClose(window))) {
        system->update(1);
        

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

    //Time calculations
    if (useCPU) {
        cpu_end = clock();
        milliseconds = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    }
    else {
        cudaEventRecord(gpu_end);
        cudaEventSynchronize(gpu_end);
        cudaEventElapsedTime(&milliseconds, gpu_start, gpu_end);

        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_end);
    }
    float it_per_sec = (float)steps * 1000 / milliseconds;
    std::cout << "Entire simulation took " << milliseconds << " ms." << std::endl;
    std::cout << "Time per iteration (ms): " << (milliseconds / (float)steps) << "." << std::endl;
    std::cout << "Iterations per second: " << it_per_sec << "." << std::endl;

    //Save to file (goes to data folder in the executable's directory
    //File is called "currentTime"_"runType"_"iterations"_
    //CurrentTime is the system time when the file was made
    //Where run type is either CPU or GPU
    //Iterations is the number of iterations used for this file
    if (saveFinal) {
        //Assumes to find a data folder in the executable's directory
        char buf[256];
        char CPU[4] = "CPU";
        char GPU[4] = "GPU";
        time_t currentTime = time(nullptr);
        char time[20];
        std::strftime(time, sizeof(time), "%Y-%m-%d_%H-%M-%S", std::localtime(&currentTime));
        if (useCPU) {
            snprintf(buf, sizeof(buf), "data/%s_%s_%d.txt", time, CPU, steps);
        }
        else {
            snprintf(buf, sizeof(buf), "data/%s_%s_%d.txt", time, GPU, steps);
        }
        
        system->writecurpostofile(buf);
        



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