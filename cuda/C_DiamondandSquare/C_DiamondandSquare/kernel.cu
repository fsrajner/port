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


__global__ void DiamondandSquare(unsigned short* deviceArray, size_t pitch, int squareSize, int arraySize, float *deviceRandom, size_t rpitch)
{

	int i = ((blockIdx.x * blockDim.x + threadIdx.x) * 2 * squareSize) + squareSize;
	int j = ((blockIdx.y * blockDim.y + threadIdx.y) * 2 * squareSize) + squareSize;
	if (i >= arraySize || j >= arraySize) return;

	unsigned short* middle = (unsigned short*)((char*)deviceArray + i * pitch) + j;
	float * rmiddle = (float*)((char*)deviceRandom + i * pitch) + j;

	unsigned short* east = (unsigned short*)((char*)deviceArray + (i)* pitch) + (j + squareSize);
	unsigned short* south = (unsigned short*)((char*)deviceArray + (i + squareSize) * pitch) + (j);
	unsigned short* west = (unsigned short*)((char*)deviceArray + (i)* pitch) + (j - squareSize);
	unsigned short* north = (unsigned short*)((char*)deviceArray + (i - squareSize) * pitch) + (j);

	float* reast = (float*)((char*)deviceRandom + (i)* pitch) + (j + squareSize);
	float* rsouth = (float*)((char*)deviceRandom + (i + squareSize) * pitch) + (j);
	float* rwest = (float*)((char*)deviceRandom + (i)* pitch) + (j - squareSize);
	float* rnorth = (float*)((char*)deviceRandom + (i - squareSize) * pitch) + (j);

	unsigned short* topleft = (unsigned short*)((char*)deviceArray + (i - squareSize) * pitch) + (j - squareSize);
	unsigned short* bottomleft = (unsigned short*)((char*)deviceArray + (i + squareSize) * pitch) + (j - squareSize);
	unsigned short* topright = (unsigned short*)((char*)deviceArray + (i - squareSize) * pitch) + (j + squareSize);
	unsigned short* bottomright = (unsigned short*)((char*)deviceArray + (i + squareSize) * pitch) + (j + squareSize);


	// generate random number rand*(largest value - smallest value + 0.999999)+smallest value
	// source: http://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
	// http://cs.brown.edu/courses/cs195v/lecture/week11.pdf
	unsigned short randomMax = (arraySize / 32);
	unsigned short randomMin = (arraySize / 64);

	//Diamond
	*middle = ((*topleft + *bottomleft + *topright + *bottomright) / 4) + unsigned short(int( (*rmiddle)*randomMax) - randomMin);

	//Square
	*north = (*topleft + *topright) / 2 + unsigned short( int( (*rnorth) * randomMax) - randomMin );
	*east = (*topright + *bottomright) / 2 + unsigned short(int((*reast) * randomMax) - randomMin );
	*south = (*bottomleft + *bottomright) / 2 + unsigned short(int((*rsouth) * randomMax) - randomMin );
	*west = (*bottomleft + *topleft) / 2 + unsigned short(int((*rwest) * randomMax) - randomMin );

}

__global__ void Smoothen(unsigned short* deviceArray, size_t pitch, int squareSize, int arraySize)
{

	int i = ((blockIdx.x * blockDim.x + threadIdx.x) * 2 * squareSize) + squareSize;
	int j = ((blockIdx.y * blockDim.y + threadIdx.y) * 2 * squareSize) + squareSize;
	if (i >= arraySize || j >= arraySize) return;

	unsigned short* middle = (unsigned short*)((char*)deviceArray + i * pitch) + j;
	
	unsigned short* east = (unsigned short*)((char*)deviceArray + (i)* pitch) + (j + 1);
	unsigned short* south = (unsigned short*)((char*)deviceArray + (i + 1) * pitch) + (j);
	unsigned short* west = (unsigned short*)((char*)deviceArray + (i)* pitch) + (j - 1);
	unsigned short* north = (unsigned short*)((char*)deviceArray + (i - 1) * pitch) + (j);


	unsigned short* topleft = (unsigned short*)((char*)deviceArray + (i - 1) * pitch) + (j - 1);
	unsigned short* bottomleft = (unsigned short*)((char*)deviceArray + (i + 1) * pitch) + (j - 1);
	unsigned short* topright = (unsigned short*)((char*)deviceArray + (i - 1) * pitch) + (j + 1);
	unsigned short* bottomright = (unsigned short*)((char*)deviceArray + (i + 1) * pitch) + (j + 1);

	//Diamond
	*middle = (*middle + (*topleft + *bottomleft + *topright + *bottomright) / 4) /2 ;

	//Square
	*north = (*north + (*topleft + *topright) / 2 ) /2;
	*east = (*east + (*topright + *bottomright) / 2) / 2;
	*south = (*south + (*bottomleft + *bottomright) / 2) / 2;
	*west = (*west + (*bottomleft + *topleft) / 2) / 2;

	*topleft = (*topleft + (*north + *east) / 2) / 2;
	*topright = (*topright + (*north + *west) / 2) / 2;
	*bottomleft = (*bottomleft + (*south + *east) / 2) / 2;
	*bottomright = (*bottomright +(*south + *west) / 2) / 2;
}

extern "C" void traverseMap(unsigned short *deviceArray, size_t pitch, int arraySize, int squareSize, int currentDepth, int maximumDepth, float *randState, size_t rpitch)
{

		dim3 threadsPerBlock(32, 32);
		dim3 numBlocks(round_div(int(pow(2, currentDepth)), threadsPerBlock.x), round_div(int(pow(2, currentDepth)), threadsPerBlock.y)); 
		DiamondandSquare << <numBlocks, threadsPerBlock >> >(deviceArray, pitch, squareSize / 2, arraySize, randState, rpitch);

		if (currentDepth < maximumDepth)	
			traverseMap(deviceArray, pitch, arraySize, squareSize / 2, currentDepth + 1, maximumDepth, randState, rpitch);
}


extern "C" void traverseWithFor(unsigned short *deviceArray, size_t pitch, int arraySize, int squareSize, int maximumDepth, float *randState, size_t rpitch)
{
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(1,1);
	int half = squareSize;
	for (int currentDepth = 0; currentDepth < maximumDepth; ++currentDepth)
	{	
		half = half / 2;
		numBlocks.x = round_div(int(pow(2, currentDepth)), threadsPerBlock.x);
		numBlocks.y = round_div(int(pow(2, currentDepth)), threadsPerBlock.y);

		DiamondandSquare << <numBlocks, threadsPerBlock >> >(deviceArray, pitch, half, arraySize, randState, rpitch);
	}

	//for (int i = 0; i < 2; ++i)
	//{
	//	Smoothen << <numBlocks, threadsPerBlock >> >(deviceArray, pitch, half, arraySize);
	//}
	cudaError err = cudaThreadSynchronize();

	/*  Check for and display Error  */
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
			__FILE__, __LINE__, cudaGetErrorString(err));
	}
}

__global__ void setValue2D(unsigned short* deviceArray, size_t pitch, int arraySize, unsigned short value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= arraySize || j >= arraySize) return;
	unsigned short* pElement = (unsigned short*)((char*)deviceArray + i * pitch) + j;
	*pElement = value;
}

extern "C" void setValuestoNull(unsigned short *deviceArray, size_t pitch, int arraySize)
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


__global__ void setupRandomKernel(float* deviceRandom, size_t pitch, int arraySize, int seed, curandState* state)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= arraySize || j >= arraySize) return;
	unsigned long long offset = i*arraySize + j;
	curand_init(offset+seed, 0 , 0, state);

	float* pElement = (float*)((char*)deviceRandom + i * pitch) + j;
	*pElement = curand_normal(state);
}

__global__ void seedKernel(unsigned short *deviceArray, size_t pitch, int arraySize)
{

	int i = 0;
	int j = 0;
	
	unsigned short* pElement = (unsigned short*)((char*)deviceArray + i * pitch) + j;
	*pElement = (arraySize / 2);
	i = arraySize - 1;
	pElement = (unsigned short*)((char*)deviceArray + i * pitch) + j;
	*pElement = (arraySize / 2);
	j = arraySize - 1;
	pElement = (unsigned short*)((char*)deviceArray + i * pitch) + j;
	*pElement = (arraySize / 2);
	i = 0;
	pElement = (unsigned short*)((char*)deviceArray + i * pitch) + j;
	*pElement = (arraySize / 2);
}

__global__ void checkValue(unsigned short* deviceArray, size_t pitch, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= size || j >= size) return;
	int* pElement = (int*)((char*)deviceArray + (i * pitch)) + j;
	if (*pElement > 0) printf("%d %d: %d\n",i,j, *pElement);
}

extern "C" void SeedandSet(unsigned short *deviceArray, size_t pitch, int arraySize, float *deviceRandom, size_t rpitch, int seed, curandState* state)
{
	seedKernel << <1, 1 >> >(deviceArray, pitch, arraySize);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(round_div(arraySize, threadsPerBlock.x), round_div(arraySize, threadsPerBlock.y));
	setupRandomKernel << <numBlocks, threadsPerBlock >> >(deviceRandom, rpitch, arraySize, seed, state);

	cudaError err = cudaThreadSynchronize();
	/*  Check for and display Error  */
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
			__FILE__, __LINE__, cudaGetErrorString(err));
	}
}