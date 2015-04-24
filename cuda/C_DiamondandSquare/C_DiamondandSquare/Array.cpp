#include "Array.h"
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
//set's the device-side array's value to 0
extern "C" void setValuestoNull(int *deviceArray, size_t pitch, int size);
//recursuce call to traverse the array
extern "C" void traverseMap(int *deviceArray, size_t pitch, int arraySize, int squareSize, int currentDepth, int maximumDepth, float *randState, size_t rpitch);
// same but wuth for cycle instead of recursive calls
extern "C" void traverseWithFor(int *deviceArray, size_t pitch, int arraySize, int squareSize, int maximumDepth, float *randState, size_t rpitch);

//this is used so that the noise can be set
extern "C" void setValuestoRandom(int *noise, int size);

//this will set a random value on the sides
extern "C" void seedandset(int *deviceArray, size_t pitch, int arraySize, float *deviceRandom, size_t rpitch, int seed);

extern "C" void asd(int *deviceArray, size_t pitch, int arraySize);


Array::Array(const int& size, const bool& recursive) : size(size), recursive(recursive)
{
	Init();

	//Traverse(recursive);
}

void Array::Init()
{
	//create host array
	hostArray = new int[size*size];

	//now we need to reserve the gpu memory as well, pitch tells the real length of one row including the padding added by the nvcc
	//http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html
	cudaMallocPitch(&deviceArray, &pitch, sizeof(int)*size, size);

	//allocate gpu memory for curand
	cudaMallocPitch(&deviceRandom, &rpitch, sizeof(float)*size, size);

	//set all values to 0
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			hostArray[i*size + j] = 0;
		}
	}
	setValuestoNull(deviceArray, pitch, size);

	//finish setting the other values
	steps = log2(size - 1);

	map = size - 1;



	seedandset(deviceArray, pitch, size, deviceRandom, rpitch, 1234);
}

void Array::Traverse(bool rec)
{
	if (rec)
		traverseMap(deviceArray, pitch, size, size, 0, steps, deviceRandom, rpitch);
	else
		traverseWithFor(deviceArray, pitch, size, size, steps, deviceRandom, rpitch);
	//asd(deviceArray, pitch, size);

	//copy it back to the host
	//cudaMemcpy2d(dst, dPitch,src ,sPitch, width, height, typeOfCopy )
	cudaMemcpy2D(hostArray, size * sizeof(int), deviceArray, pitch, size * sizeof(int), size, cudaMemcpyDeviceToHost);

}

void Array::setColor(IplImage* im, int x, int y, UCHAR r, UCHAR g, UCHAR b)
{
	im->imageData[y*im->widthStep + 3 * x + 2] = r;
	im->imageData[y*im->widthStep + 3 * x + 1] = g;
	im->imageData[y*im->widthStep + 3 * x] = b;
}

void Array::createImage(IplImage* image)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			setColor(image, i, j, UCHAR(int((double(hostArray[size*i + j]) / double(size)) * 255)), UCHAR(int((double(hostArray[size*i + j]) / double(size)) * 255)), UCHAR(int((double(hostArray[size*i + j]) / double(size)) * 255)));
		}
	}
	}


Array::~Array()
{
	delete[] hostArray;
	cudaFree(deviceRandom);
	cudaFree(deviceArray);

}
