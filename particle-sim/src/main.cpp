// Standard Library Includes
#include <stdio.h>
#include <ctime>
#include <iostream>

// Program Includes
#include <common.h>
#include <shaderClass.h>
#include <buffers.h>
#include <particleSystem.h>
#include <particleSystemCpu.h>
#include <particleSystemGpu.h>

// Visualization Includes
#if (RENDER_ENABLE)
#include <glad/glad.h>
#include <GLFW/glfw3.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height){
  glViewport(0, 0, width, height);
}
#endif

#if (GPU_ENABLE)
// GPU Library Includes
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

int main(int argc, char** argv) {

    int steps = 0;

    ParticleSystem* system;

    //Set up timers
#if (TIMING_ENABLE)
#if (GPU_ENABLE)
    cudaEvent_t gpu_start, gpu_end;
#else
    clock_t cpu_start, cpu_end;
#endif
    float milliseconds;
#endif

#if (RENDER_ENABLE)
    GLFWwindow* window;

    //Initialize GLFW 
    glfwInit();

    //This window is where we will view our graphics
    // (width, height, title, monitor, share)
    window = glfwCreateWindow(width, height, "Particle Simulation", NULL, NULL);

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
#else
    if (max_steps < 0) {
      printf("Warning: Can only exit program via ctrl-c. This is not recommended unless testing code.\n");
    }
#endif

#if (GPU_ENABLE)
    system = new ParticleSystemGPU(numParticles, systemInitType, seed);
#if (TIMING_ENABLE)
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);
#endif
#else
    system = new ParticleSystemCPU(numParticles, systemInitType, seed);
#if (TIMING_ENABLE)
    cpu_start = clock();
#endif
#endif
    
    //This loop runs until the window is closed (or I guess if we make the program exit somehow)
    while (steps != max_steps) {
        system->update(1e-15);

        steps++;

#if (RENDER_ENABLE)
        if (glfwWindowShouldClose(window)) {
          break;
        }

        system->display();

        //Swap the buffers so we see the new frame generated.
        glfwSwapBuffers(window);

        //We want to draw the points onto the screen:
        glfwPollEvents();
#endif
    }

#if (TIMING_ENABLE)
#if (GPU_ENABLE)
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&milliseconds, gpu_start, gpu_end);
#else
    cpu_end = clock();
    milliseconds = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
#endif

    float it_per_sec = (float)steps * 1000 / milliseconds;
    std::cout << "Entire simulation took " << milliseconds << " ms." << std::endl;
    std::cout << "Time per iteration (ms): " << (milliseconds / (float)steps) << "." << std::endl;
    std::cout << "Iterations per second: " << it_per_sec << "." << std::endl;

#endif


#if (SAVE_FINAL)
    //Save to file (goes to data folder in the executable's directory
    //File is called "currentTime"_"runType"_"iterations"_
    //CurrentTime is the system time when the file was made
    //Where run type is either CPU or GPU
    //Iterations is the number of iterations used for this file
 
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
#endif
    
  //We have completed so we need to clean up.
#if (RENDER_ENABLE)
  //Terminate GLFW
  glfwDestroyWindow(window);
  glfwTerminate();
#endif

#if (GPU_ENABLE && TIMING_ENABLE)
  cudaEventDestroy(gpu_start);
  cudaEventDestroy(gpu_end);
#endif

  //Delete the particleSystem
  delete system;
    
  return 0;

}