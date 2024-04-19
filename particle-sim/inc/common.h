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
constexpr int max_steps = 1000; //Cutoff number of iterations, this is handy if rendering is false to determine a stop. Set to -1 to never terminate
constexpr int seed = 42; //Seed for run, set to 1 for random generation.

// Physical parameters - mass
constexpr float proton_mass = 1.0f; //This is in atomic mass units 1amu ~ 1.67e-27 kg.
constexpr float electron_mass = 0.00055f;
constexpr float mass_ratio = 1836.15267343f; //This is the ratio with uncertainty of 6.0e-11.

// More Constants :)
constexpr double inv_masses[] = { 1.09776e30, 5.978638e26, 5.978638e26 };
constexpr float charges[] = { -1, 1, 0 };

// Physical parameters - forces
constexpr float residual_strong_range = 3; //In fentometers (10^-15 m)

//refer to equations.ipynb to see why these value is what it is.
constexpr float coulomb_scalar = 2.310272969e-10; //N*nanometers^2
constexpr float yukawa_scalar = 1.9692204e-9;    //Experimentally obtained
constexpr float yukawa_radius = 1.4e-3;			 //Radius of strength.
constexpr float yukawa_cutoff = 1e-3;          //Sweet spot. (Strong force likes to be between 0.8 and 1.4 fm.

#endif