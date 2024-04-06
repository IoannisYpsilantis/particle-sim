# n-body Particle Simulation

This n-body simulation is designed to simulate the motion of protons and electrons in order to visualize higher order properties of atoms. \

Visualization is processed using OpenGL via glad. All GPU functions were developed using CUDA. 


## Installation

This project is built using Visual Studio 2019. Dependencies include:

[Cuda](https://developer.nvidia.com/cuda-11-6-0-download-archive)

[Glad](https://glad.dav1d.de)

[GLFW](https://www.glfw.org/download.html)

Follow [Victor Gordan's OpenGl Tutorial](https://www.youtube.com/watch?v=XpBGwZNyUh0&list=PLPaoO-vpZnumdcb4tZc4x5Q-v7CkrQ6M-&index=1) on installation process for Glad and GLFW. Note that instead of a blank C++ project. We used the Cuda solution for Visual Studio. Rather than installing The Libraries folder within the src directory, Libraries should be stored in:

C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Libraries\particle-sim\include

and

C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Libraries\particle-sim\lib

If you want to change this setup, the visual studio project must be reconfigured to reflect changes.
