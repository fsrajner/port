#pragma once
#include <math.h>
#include <random>
#include <iostream>
//cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdlib.h>
#include <stdint.h>
#include <png.h>
#include "png++\png.hpp"

class Array
{
private:
	// array containing the values host-side
	unsigned short* hostArray;

	// values for device-side array
	size_t pitch;
	unsigned short* deviceArray;
	// CUDA pointer for random numbers and pitch
	float *deviceRandom;
	size_t rpitch;
	//size of the array. The volume should be considered as a cube, the values will be later scaled to 0 as min, and 65535 as max
	int size;

	//last index of the array
	int map;

	//number of recursive steps n in 2^n +1=size
	int steps;

	bool recursive;

	curandState* state;

public:
	void Traverse(bool rec);
	Array(const size_t& size = 17, const bool& recursive = false);
	void createImage();
	~Array();
};

