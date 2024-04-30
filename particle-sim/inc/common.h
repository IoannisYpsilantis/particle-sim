#ifndef __COMMON_H_
#define __COMMON_H_

//This is added to the file name - very helpful for associating runs for comparison
constexpr int ID = 5;

// Program Controls
#define RENDER_ENABLE (0)
#define TIMING_ENABLE (1)
#define GPU_ENABLE (1)
#define SAVE_FINAL (1)
						

//Program Execution Type Specification
#define doubleBuffer (0) //Whether to double buffer positions or not.
#define binningGPU (0) //False is naive
						//True is to be implemented improved GPU. (Might be handy to make this a control, it also might become an int)

// OpenGL Window Parameters
constexpr int width = 800;
constexpr int height = 800;

constexpr int ELECTRON_COLOR[3] = { 0, 180, 255 }; // Blue
constexpr int PROTON_COLOR[3] = { 255, 0, 0 }; // Red
constexpr int NEUTRON_COLOR[3] = { 204, 204, 0 }; // Yellow

// Environment Parameters
constexpr int numParticles = 500;
constexpr int systemInitType = 2;
constexpr int maxSteps = 100; //Cutoff number of iterations, this is handy if rendering is false to determine a stop. Set to -1 to never terminate
constexpr int seed = 42; //Seed for run, set to 1 for random generation.
constexpr float timeStep = 1e-7;
constexpr float dampingFactor = 0.999;
constexpr int boundingBox = 200000; //The size of the problem

//Given timeStep is 1e-7 it seems have numParticles = 5 * boundingBox is pretty good.

// More Constants :)
constexpr float inv_masses[] = { 1.836152673e3, 1.0, 1.0 }; //This is in AU (1AU ~ 1 proton ~ 1 nuetron.
constexpr float charges[] = { -1.0, 1.0, 0.0 };

//refer to equations.ipynb to see why these value is what it is. 
constexpr float distance_step = 1e-15; //The step of position. Used for some initializiation

constexpr float coulomb_scalar = 1.381228459 * 1e-1/distance_step;  //AU*picometers^3/second^2
constexpr float yukawa_scalar = 1.177325499  * 1/distance_step;     //Experimentally obtained - AU*picometers^3/second^2
constexpr float yukawa_radius = 1.4          * 1e-15/distance_step;	//Radius of strength.
constexpr float yukawa_cutoff = 0.8          * 1e-15/distance_step; //Sweet spot. (Strong force likes to be between 0.8 and 1.4 fm.


// GPU Configuration
#define TILE_SIZE 128

#if binningGPU
//How many bins in each direction - could define based on env (bounding box)
constexpr int binWidth = 100;
constexpr int binHeight = 100;
constexpr int binDepth = 100;
constexpr int CoulombRadius = 15; //Radius of bins for Coulomb forces
constexpr int YukawaRadius = 1;   //Radius of bins for Yukawa forces

//If radius is too small then the algorithm is incorrect
//If the radius is too big then more is calculated than necessary
#endif

#endif