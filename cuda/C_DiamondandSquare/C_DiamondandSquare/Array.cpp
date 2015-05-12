#include "Array.h"
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
//set's the device-side array's value to 0
extern "C" void setValuestoNull(unsigned short *deviceArray, size_t pitch, int size);
//recursuce call to traverse the array
extern "C" void traverseMap(unsigned short *deviceArray, size_t pitch, int arraySize, int squareSize, int currentDepth, int maximumDepth, float *randState, size_t rpitch);
// same but wuth for cycle instead of recursive calls
extern "C" void traverseWithFor(unsigned short *deviceArray, size_t pitch, int arraySize, int squareSize, int maximumDepth, float *randState, size_t rpitch);

//this is used so that the noise can be set
extern "C" void setValuestoRandom(int *noise, int size);

//this will set a random value on the sides
extern "C" void SeedandSet(unsigned short *deviceArray, size_t pitch, int arraySize, float *deviceRandom, size_t rpitch, int seed, curandState* state);

extern "C" void asd(unsigned short *deviceArray, size_t pitch, int arraySize);


Array::Array(const size_t& size, const bool& recursive) : size(size), recursive(recursive)
{
	//create host array
	hostArray = new unsigned short[size*size];

	//now we need to reserve the gpu memory as well, pitch tells the real length of one row including the padding added by the nvcc
	//http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html
	cudaMallocPitch(&deviceArray, &pitch, sizeof(unsigned short)*size, size);

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

	state = new curandState{};

	SeedandSet(deviceArray, pitch, size, deviceRandom, rpitch, 1234, state);
}



void Array::Traverse(bool rec)
{
	if (rec)
		traverseMap(deviceArray, pitch, size, size, 0, steps, deviceRandom, rpitch);
	else
		traverseWithFor(deviceArray, pitch, size, size, steps, deviceRandom, rpitch);

	//copy it back to the host
	cudaMemcpy2D(hostArray, size * sizeof(unsigned short), deviceArray, pitch, size * sizeof(unsigned short), size, cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (hostArray[size*i + j] != 0)
				std::cout << i << " " << j << ": " << hostArray[size*i + j] << '\n';
		}
	}*/

}

void abort_(const char * s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr, "\n");
	va_end(args);
	abort();
}

void Array::createImage()
{

	png::image< png::gray_pixel_16 > image(size, size);
	
	for (size_t y = 0; y < image.get_height(); ++y)
	{
		for (size_t x = 0; x < image.get_width(); ++x)
		{
			// copy 16-bit values from array into pixel
			png::gray_pixel_16 pix = unsigned short(static_cast<int>((hostArray[size*x + y] / static_cast<double>(size)) * 65535));

			// draw 16-bit values to image
			image[y][x] = pix;
		}
	}
	image.write("rgb.png");
}



Array::~Array()
{
	delete[] hostArray;
	cudaFree(deviceRandom);
	cudaFree(deviceArray);

}
