#ifndef __COMMON_H_
#define __COMMON_H_

// Program Controls
#define RENDER_ENABLE (1)
#define TIMING_ENABLE (0)
#define GPU_ENABLE (1)
#define SAVE_FINAL (0)

// GPU Configuration
#define TILE_SIZE 128

// OpenGL Window Parameters
constexpr int width = 800;
constexpr int height = 800;

constexpr int ELECTRON_COLOR[3] = { 0, 180, 255 }; // Blue
constexpr int PROTON_COLOR[3] = { 255, 0, 0 }; // Red
constexpr int NEUTRON_COLOR[3] = { 204, 204, 0 }; // Yellow

// Environment Parameters
constexpr int numParticles = 1000;
constexpr int systemInitType = 2;
constexpr int max_steps = -1; //Cutoff number of iterations, this is handy if rendering is false to determine a stop. Set to -1 to never terminate
constexpr int seed = 42; //Seed for run, set to 1 for random generation.

// More Constants :)
constexpr float inv_masses[] = { 1.836152673e3, 1.0, 1.0 }; //This is in AU (1AU ~ 1 proton ~ 1 nuetron.
constexpr float charges[] = { -1.0, 1.0, 0.0 };

//refer to equations.ipynb to see why these value is what it is. 
constexpr float coulomb_scalar = 1.381228459e8; //AU*nanometers^3/second^2
constexpr float yukawa_scalar = 1.177325499e9;    //Experimentally obtained - AU*nanometers^3/second^2
constexpr float yukawa_radius = 1.4e-6;			 //Radius of strength.
constexpr float yukawa_cutoff = 1e-6;          //Sweet spot. (Strong force likes to be between 0.8 and 1.4 fm.

#endif