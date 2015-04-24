#pragma once
#include <math.h>
#include <random>
#include <iostream>
//cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

class Array
{
private:
	// array containing the values host-side
	int* hostArray;

	// values for device-side array
	size_t pitch;
	int* deviceArray;
	// CUDA pointer for random numbers
	float *deviceRandom;
	size_t rpitch;
	//size of the array. The volume should be considered as a cube, the values will be later scaled to 0 as min, and 255 as max
	int size;

	//last index of the array
	int map;

	//number of recursive steps n in 2^n +1=size
	int steps;

	bool recursive;

	void Init();

public:
	void Traverse(bool rec);
	Array(const int& size = 17, const bool& recursive = false);
	void createImage(IplImage* image);
	void setColor(IplImage* im, int x, int y, UCHAR r, UCHAR g, UCHAR b);
	~Array();
};

