// includes, system
#include <iostream>
#include <stdlib.h>

#include <chrono>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>

#include "Array.h"
#include "PUDaS.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	// input data, size of texture
	size_t len = 2049;
	Array array{ len };

	//CPUDaS<int> cpu{ len };
	//for (int i = 0; i < cpu.getNumberofSteps(); i++)
	//{
	//	cpu.getnoise()[i] = (len / ((i * 2) - i + 1));
	//}

	
	std::cout << "CPU START : \n";

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	//cpu.traverse();
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();

	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	std::cout << "CPU END : " << millis  << '\n';
	
	std::cout << "GPU recursive START : \n";

	start = std::chrono::system_clock::now();
	//array.Traverse(true);
	end = std::chrono::system_clock::now();

	millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "GPU recursive END : " << millis << '\n';

	std::cout << "GPU START : \n";

	start = std::chrono::system_clock::now();
	array.Traverse(false);
	end = std::chrono::system_clock::now();

	millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "GPU END : " << millis << '\n';



	array.createImage();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
