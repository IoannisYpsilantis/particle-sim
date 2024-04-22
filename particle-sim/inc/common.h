#ifndef __COMMON_H_
#define __COMMON_H_

//This is added to the file name - very helpful for associating runs for comparison
constexpr int ID = 1;

// Program Controls
#define RENDER_ENABLE (1)
#define TIMING_ENABLE (1)
#define GPU_ENABLE (1)
#define SAVE_FINAL (0)
						

//Program Execution Type Specification
#define advancedGPU (0) //False is naive
						//True is to be implemented improved GPU. (Might be handy to make this a control, it also might become an int)

// GPU Configuration
#define TILE_SIZE 128

// OpenGL Window Parameters
constexpr int width = 800;
constexpr int height = 800;

constexpr int ELECTRON_COLOR[3] = { 0, 180, 255 }; // Blue
constexpr int PROTON_COLOR[3] = { 255, 0, 0 }; // Red
constexpr int NEUTRON_COLOR[3] = { 204, 204, 0 }; // Yellow

// Environment Parameters
constexpr int numParticles = 10000;
constexpr int systemInitType = 2;
constexpr int maxSteps = -1; //Cutoff number of iterations, this is handy if rendering is false to determine a stop. Set to -1 to never terminate
constexpr int seed = 42; //Seed for run, set to 1 for random generation.
constexpr float timeStep = 1e-7;
constexpr float dampingFactor = 0.999;
constexpr int boundingBox = 1000; //The size of the problem NOTE* NEED TO UPDATE shaderclass.cpp if you change this value!

// More Constants :)
constexpr float inv_masses[] = { 1.836152673e3, 1.0, 1.0 }; //This is in AU (1AU ~ 1 proton ~ 1 nuetron.
constexpr float charges[] = { -1.0, 1.0, 0.0 };

//refer to equations.ipynb to see why these value is what it is. 
constexpr float coulomb_scalar = 1.381228459e11; //AU*picometers^3/second^2
constexpr float yukawa_scalar = -1.177325499e12;    //Experimentally obtained - AU*picometers^3/second^2
constexpr float yukawa_radius = 1.4e-3;			 //Radius of strength.
constexpr float yukawa_cutoff = 0.8e-3;          //Sweet spot. (Strong force likes to be between 0.8 and 1.4 fm.

constexpr float distance_step = 1e-12; //The step of position, this doesn't do anything just helpful to see what's happening.
#endif