#include <stdio.h>
#include <math.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>

#include "shaderClass.h"




struct particle_pos {
    GLfloat x;
    GLfloat y;
    GLfloat z;
    
} typedef Particle_pos;

struct particle_col {
    GLuint r;
    GLuint g;
    GLuint b;
} typedef Particle_col;
int main(int argc, char** argv) {

    //Initialize particles
    Particle_pos particles_pos[1000];
    Particle_col particles_col[1000];

    for (int i = 0; i < 1000; i++) {
        float theta = (float) (999 - i) / 1000 * 2 * 3.1415;
        particles_pos[i].x = (GLfloat) cos(theta);
        particles_pos[i].y = (GLfloat) sin(theta);
        particles_pos[i].z = 1;
        particles_col[i].r = i % 255;
        particles_col[i].g = 255 - (i % 255);
        particles_col[i].b = 55;


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



    Shader shaderProgram;



    // VBO (Vertex Buffer Object) -> Buffer to send stuff to the GPU
    // VAO (Vertex Array Object) -> Stores pointers to VBOs and tells openGL how to interpret the data
    GLuint VAO, positionBuffer, colorBuffer;

    //One VBO so we put 1. 
    glGenVertexArrays(1, &VAO);

    glBindVertexArray(VAO);

    glGenBuffers(1, &positionBuffer);
    glGenBuffers(1, &colorBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(particles_pos), particles_pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(particles_col), particles_col, GL_STATIC_DRAW);
    glVertexAttribIPointer(1, 3, GL_UNSIGNED_INT, 0, (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);


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
    //glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    //glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(3*sizeof(GLfloat)));

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
        float temp_x = particles_pos[0].x;
        float temp_y = particles_pos[0].y;
        float temp_z = particles_pos[0].z; 

        for (int i = 0; i < 999; i++) {
            particles_pos[i].x = particles_pos[i + 1].x;
            particles_pos[i].y = particles_pos[i + 1].y;
            particles_pos[i].z = particles_pos[i + 1].z;
        }

        particles_pos[999].x = temp_x;
        particles_pos[999].y = temp_y;
        particles_pos[999].z = temp_z;

        glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(particles_pos), particles_pos, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    //We have completed so we need to clean up.

    //Delete the VBO and VAO
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &positionBuffer);
    glDeleteBuffers(1, &colorBuffer);

    //Delete the shader program
    shaderProgram.Delete();

    //Terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

}