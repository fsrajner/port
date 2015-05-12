Diamond and Square on CUDA



This program integrates CUDA into an existing C++ application, i.e. the CUDA entry point on host side is only a function which is called from C++ code and only the file containing this function is compiled with nvcc.

It produces a procedural 16 bit greyscale png heightmap.