// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <random>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// random stuff
#include <curand.h>
#include <curand_kernel.h>

//my helper functions
unsigned int round_div(unsigned int dividend, unsigned int divisor)
{
	return (dividend + divisor - 1) / divisor;
}



__global__ void Diamond(int* deviceArray, size_t pitch, int diamondSize, int currentDepth, int arraySize, float *state)
{
	int i = ((blockIdx.x * blockDim.x + threadIdx.x) * 2 * diamondSize) +diamondSize;
	int j = ((blockIdx.y * blockDim.y + threadIdx.y) * 2 * diamondSize) +diamondSize;
	if (i >= arraySize || j >= arraySize) return;

	
	int* a = (int*)((char*)deviceArray + (i - diamondSize) * pitch) + (j - diamondSize);
	int* b = (int*)((char*)deviceArray + (i + diamondSize) * pitch) + (j - diamondSize);
	int* c = (int*)((char*)deviceArray + (i - diamondSize) * pitch) + (j + diamondSize);
	int* d = (int*)((char*)deviceArray + (i + diamondSize) * pitch) + (j + diamondSize);

	int value = ((*a + *b + *c + *d) / 4);

	int* pElement = (int*)((char*)deviceArray + i * pitch) + j;
	*pElement = value;
}

__global__ void Square(int* deviceArray, size_t pitch, int squareSize, int currentDepth, int arraySize, float *state)
{

	int i = ((blockIdx.x * blockDim.x + threadIdx.x) * 2 * squareSize) + squareSize;
	int j = ((blockIdx.y * blockDim.y + threadIdx.y) * 2 * squareSize) + squareSize;
	if (i >= arraySize || j >= arraySize) return;

	int* east = (int*)((char*)deviceArray + (i)* pitch) + (j + squareSize);
	int* south = (int*)((char*)deviceArray + (i + squareSize) * pitch) + (j);
	int* west = (int*)((char*)deviceArray + (i)* pitch) + (j - squareSize);
	int* north = (int*)((char*)deviceArray + (i - squareSize) * pitch) + (j);

	int* topleft = (int*)((char*)deviceArray + (i - squareSize) * pitch) + (j - squareSize);
	int* bottomleft = (int*)((char*)deviceArray + (i + squareSize) * pitch) + (j - squareSize);
	int* topright = (int*)((char*)deviceArray + (i - squareSize) * pitch) + (j + squareSize);
	int* bottomright = (int*)((char*)deviceArray + (i + squareSize) * pitch) + (j + squareSize);

	*north = (*topleft + *topright) / 2;
	*east = (*topright + *bottomright) / 2;
	*south = (*bottomleft + *bottomright) / 2;
	*west = (*bottomleft + *topleft) / 2;
}

__global__ void DiamondandSquare(int* deviceArray, size_t pitch, int squareSize, int currentDepth, int arraySize, float *deviceRandom, size_t rpitch)
{

	int i = ((blockIdx.x * blockDim.x + threadIdx.x) * 2 * squareSize) + squareSize;
	int j = ((blockIdx.y * blockDim.y + threadIdx.y) * 2 * squareSize) + squareSize;
	if (i >= arraySize || j >= arraySize) return;

	int* middle = (int*)((char*)deviceArray + i * pitch) + j;
	float * rmiddle = (float*)((char*)deviceRandom + i * pitch) + j;

	int* east = (int*)((char*)deviceArray + (i)* pitch) + (j + squareSize);
	int* south = (int*)((char*)deviceArray + (i + squareSize) * pitch) + (j);
	int* west = (int*)((char*)deviceArray + (i)* pitch) + (j - squareSize);
	int* north = (int*)((char*)deviceArray + (i - squareSize) * pitch) + (j);

	float* reast = (float*)((char*)deviceRandom + (i)* pitch) + (j + squareSize);
	float* rsouth = (float*)((char*)deviceRandom + (i + squareSize) * pitch) + (j);
	float* rwest = (float*)((char*)deviceRandom + (i)* pitch) + (j - squareSize);
	float* rnorth = (float*)((char*)deviceRandom + (i - squareSize) * pitch) + (j);

	int* topleft = (int*)((char*)deviceArray + (i - squareSize) * pitch) + (j - squareSize);
	int* bottomleft = (int*)((char*)deviceArray + (i + squareSize) * pitch) + (j - squareSize);
	int* topright = (int*)((char*)deviceArray + (i - squareSize) * pitch) + (j + squareSize);
	int* bottomright = (int*)((char*)deviceArray + (i + squareSize) * pitch) + (j + squareSize);


	// generate random number rand*(largest value - smallest value + 0.999999)+smallest value
	// source: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
	// http://cs.brown.edu/courses/cs195v/lecture/week11.pdf
	int randomMax = (arraySize / 8);
	int randomMin = -(arraySize / 16);

	//Diamond
	*middle = ((*topleft + *bottomleft + *topright + *bottomright) / 4) + int((*rmiddle)*randomMax + randomMin);

	//Square
	*north = (*topleft + *topright) / 2 + int( (*rnorth) * randomMax + randomMin);
	*east = (*topright + *bottomright) / 2 + int( (*reast) * randomMax + randomMin);
	*south = (*bottomleft + *bottomright) / 2 + int( (*rsouth) * randomMax + randomMin);
	*west = (*bottomleft + *topleft) / 2 + int( (*rwest) * randomMax + randomMin);
	/*
	//Diamond
	*middle = int(truncf((*rmiddle)*randomMax) + randomMin);

	//Square
	*north = int(truncf((*rnorth) * randomMax) + randomMin);
	*east = int(truncf((*reast) * randomMax) + randomMin);
	*south = int(truncf((*rsouth) * randomMax) + randomMin);
	*west = int(truncf((*rwest) * randomMax) + randomMin);
	*/
}

extern "C" void traverseMap(int *deviceArray, size_t pitch, int arraySize, int squareSize, int currentDepth, int maximumDepth, float *randState, size_t rpitch)
{

		dim3 threadsPerBlock(32, 32);
		dim3 numBlocks(round_div(int(pow(2, currentDepth)), threadsPerBlock.x), round_div(int(pow(2, currentDepth)), threadsPerBlock.y)); 
		DiamondandSquare << <numBlocks, threadsPerBlock >> >(deviceArray, pitch, squareSize / 2, currentDepth, arraySize, randState, rpitch);
		/*
		cudaError err = cudaThreadSynchronize();

		//  Check for and display Error  
		if (cudaSuccess != err)
		{
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
				__FILE__, __LINE__, cudaGetErrorString(err));
		}
		*/
		if (currentDepth < maximumDepth)	
			traverseMap(deviceArray, pitch, arraySize, squareSize / 2, currentDepth + 1, maximumDepth, randState, rpitch);
}

__global__ void checkseed(int* deviceArray, size_t pitch, int arraySize)
{
	int i = 0;
	int j = 0;
	printf("asd\n");
	int* pElement = (int*)((char*)deviceArray + i * pitch) + j;
	printf("%d\n", *pElement);
	i = arraySize - 1;
	pElement = (int*)((char*)deviceArray + i * pitch) + j;
	printf("%d\n", *pElement);
	j = arraySize - 1;
	pElement = (int*)((char*)deviceArray + i * pitch) + j;
	printf("%d\n", *pElement);
	i = 0;
	pElement = (int*)((char*)deviceArray + i * pitch) + j;
	printf("%d\n", *pElement);
}
extern "C" void asd(int *deviceArray, size_t pitch, int arraySize)
{
	checkseed << <1, 1 >> >(deviceArray, pitch, arraySize);
}

extern "C" void traverseWithFor(int *deviceArray, size_t pitch, int arraySize, int squareSize, int maximumDepth, float *randState, size_t rpitch)
{
	dim3 threadsPerBlock(32, 32);
	int half = squareSize;
	for (int currentDepth = 0; currentDepth < maximumDepth; currentDepth++)
	{	
		
		half = half / 2;
		
		dim3 numBlocks(round_div(int(pow(2, currentDepth)), threadsPerBlock.x), round_div(int(pow(2, currentDepth)), threadsPerBlock.y));
		if (currentDepth == maximumDepth - 1) printf("%d %d\n", numBlocks.x, numBlocks.y);
		DiamondandSquare << <numBlocks, threadsPerBlock >> >(deviceArray, pitch, half, currentDepth, arraySize, randState, rpitch);
		

		
	}
	cudaError err = cudaThreadSynchronize();

	/*  Check for and display Error  */
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
			__FILE__, __LINE__, cudaGetErrorString(err));
	}
}

__global__ void setValue2D(int* deviceArray, size_t pitch, int size, int value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= size || j >= size) return;
	int* pElement = (int*)((char*)deviceArray + i * pitch) + j;
	*pElement = value;
}

extern "C" void setValuestoNull(int *deviceArray, size_t pitch, int arraySize)
{
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(round_div(arraySize, threadsPerBlock.x), round_div(arraySize, threadsPerBlock.y));

	setValue2D << <numBlocks, threadsPerBlock >> >(deviceArray, pitch, arraySize, 0);
}


__global__ void setValue1D(int* noise, int arraySize)
{
	for (int i = 0; i < arraySize; i++)
	{
		noise[i] = (arraySize / ((i * 2) - i + 1));
	}
}


__global__ void setupRandomKernel(float* deviceRandom, size_t pitch, int arraySize, int seed)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= arraySize || j >= arraySize) return;
	curandState state;
	unsigned long long offset = i*arraySize + j;
	curand_init(offset+seed, 0 , 0, &state);

	float* pElement = (float*)((char*)deviceRandom + i * pitch) + j;
	*pElement = curand_normal(&state);
}

__global__ void seedKernel(int *deviceArray, size_t pitch, int arraySize)
{

	int i = 0;
	int j = 0;
	
	int* pElement = (int*)((char*)deviceArray + i * pitch) + j;
	*pElement = arraySize / 2;
	i = arraySize - 1;
	pElement = (int*)((char*)deviceArray + i * pitch) + j;
	*pElement = arraySize / 2;
	j = arraySize - 1;
	pElement = (int*)((char*)deviceArray + i * pitch) + j;
	*pElement = arraySize / 2;
	i = 0;
	pElement = (int*)((char*)deviceArray + i * pitch) + j;
	*pElement = arraySize / 2;
}

__global__ void checkValue(int* deviceArray, size_t pitch, int size, int value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= size || j >= size) return;
	int* pElement = (int*)((char*)deviceArray + (i * pitch)) + j;
	if (*pElement > 0) printf("%d %d: %d\n",i,j, *pElement);
}



extern "C" void seedandset(int *deviceArray, size_t pitch, int arraySize, float *deviceRandom, size_t rpitch, int seed)
{
	seedKernel << <1, 1 >> >(deviceArray, pitch, arraySize);
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(round_div(arraySize, threadsPerBlock.x), round_div(arraySize, threadsPerBlock.y));
	setupRandomKernel << <numBlocks, threadsPerBlock >> >(deviceRandom, rpitch, arraySize, seed);
	//cudaDeviceSynchronize();
	cudaError err = cudaThreadSynchronize();

	/*  Check for and display Error  */
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
			__FILE__, __LINE__, cudaGetErrorString(err));
	}
	//dim3 threadsPerBlock(16, 16);
	//dim3 numBlocks(round_div(arraySize, threadsPerBlock.x), round_div(arraySize, threadsPerBlock.y));

	//checkValue << <numBlocks, threadsPerBlock >> >(deviceArray, pitch, arraySize, 0);
}