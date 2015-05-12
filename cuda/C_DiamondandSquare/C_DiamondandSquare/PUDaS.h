#pragma once
#include <math.h>
#include <random>
#include <iostream>


	template <typename T>
class CPUDaS
{
private:
	// array containing the values
	T** mapvalues;

	//size of the array. it should be considered as a cube, the values will be later scaled to 0 min, and 255 max
	int size;

	//last index of array
	int map;

	//number of recursive steps
	int steps;

	//array containing noise
	T* noise;

	//random generator
	std::random_device rd;

public:
	CPUDaS(const int& size = 0);
	~CPUDaS();

	int getNumberofSteps();
	T* getnoise();

	void seed();

	void list(FILE* im);

	void traverse();

	//void setColor(FILE* im, int x, int y, UCHAR r, UCHAR g, UCHAR b);
private:
	void square(int x, int y, int size, int currentdepth); // center position, and size of square
	void diamond(int x, int y, int size, int currentdepth); // center position, and size of diamond
	void traverseMap(int size, int currentdepth); // traversing thhrough the array
	T getvalue(int x, int y);
	T average(T a, T b, T c, T d);
};


template <typename T>
CPUDaS<T>::CPUDaS(const int& s) : size(s)
{
	mapvalues = new T*[size];
	for (int i = 0; i < size; i++)
		mapvalues[i] = new T[size];

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			mapvalues[j][i] = 0;
		}
	}

	steps = log2(size - 1) + 1;

	noise = new T[steps];

	map = size - 1;
}

template <typename T>
CPUDaS<T>::~CPUDaS()
{
	for (int i = 0; i < size; ++i)
		delete[] mapvalues[i];
	delete[] mapvalues;
}

template <typename T>
int CPUDaS<T>::getNumberofSteps()
{
	return steps;
}

template <typename T>
T* CPUDaS<T>::getnoise()
{
	return noise;
}

template <typename T>
void CPUDaS<T>::seed()
{

	mapvalues[0][0] = noise[0] + (rd() % (noise[0] / 2));
	mapvalues[0][size - 1] = noise[0] + (rd() % (noise[0] / 2));
	mapvalues[size - 1][0] = noise[0] + (rd() % (noise[0] / 2));
	mapvalues[size - 1][size - 1] = noise[0] + (rd() % (noise[0] / 2));
}

template <typename T>
void CPUDaS<T>::square(int x, int y, int squaresize, int currentdepth)
{
	T topleft = getvalue(x - squaresize, y + squaresize);
	T topright = getvalue(x + squaresize, y + squaresize);
	T botleft = getvalue(x - squaresize, y - squaresize);
	T botright = getvalue(x + squaresize, y - squaresize);

	mapvalues[y + squaresize][x] = (topleft + topright) / 2 + (rd() % (noise[currentdepth] / 2));
	mapvalues[y - squaresize][x] = (botleft + botright) / 2 + (rd() % (noise[currentdepth] / 2));
	mapvalues[y][x + squaresize] = (botright + topright) / 2 + (rd() % (noise[currentdepth] / 2));
	mapvalues[y][x - squaresize] = (topleft + botleft) / 2 + (rd() % (noise[currentdepth] / 2));
}

template <typename T>
void CPUDaS<T>::diamond(int x, int y, int squaresize, int currentdepth)
{

	T value = average(
		getvalue(x - squaresize, y + squaresize),
		getvalue(x + squaresize, y + squaresize),
		getvalue(x - squaresize, y - squaresize),
		getvalue(x + squaresize, y - squaresize)
		) + (rd() % (noise[currentdepth] / 2));

	//	value = T(	(double(value)/double(size)*	255);

	mapvalues[y][x] = value;
}

template <typename T>
void CPUDaS<T>::traverseMap(int squaresize, int currentdepth)
{
	int half = squaresize / 2;


	if (currentdepth == steps) return;

	for (int y = half; y < map; y += squaresize)
	{
		for (int x = half; x < map; x += squaresize)
		{
			square(x, y, half, currentdepth);
		}
	}


	for (int y = 0; y <= map; y += half) {
		for (int x = (y + half) % squaresize; x <= map; x += squaresize) {
			diamond(x, y, half, currentdepth);
		}
	}

	traverseMap(half, currentdepth + 1);


}

template <typename T>
void CPUDaS<T>::traverse()
{
	seed();

	traverseMap(map, 1);
}

template <typename T>
T CPUDaS<T>::average(T a, T b, T c, T d)
{
	return ((a + b + c + d) / 4);
}


/*
template <typename T>
void CPUDaS<T>::setColor(FILE* im, int x, int y, UCHAR r, UCHAR g, UCHAR b)
{
	im->imageData[y*im->widthStep + 3 * x + 2] = r;
	im->imageData[y*im->widthStep + 3 * x + 1] = g;
	im->imageData[y*im->widthStep + 3 * x] = b;
}
*/
template <typename T>
void CPUDaS<T>::list(FILE* im)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			std::cout << int((double(mapvalues[j][i]) / double(size)) * 255) << " ";
			if (j + 1 == size) std::cout << std::endl;
			setColor(im, j, i, UCHAR(int((double(mapvalues[j][i]) / double(size)) * 255)), UCHAR(int((double(mapvalues[j][i]) / double(size)) * 255)), UCHAR(int((double(mapvalues[j][i]) / double(size)) * 255)));
		}
	}
}

template <typename T>
T CPUDaS<T>::getvalue(int x, int y)
{
	if (x<0 || x>size - 1) return 0;
	if (y<0 || y>size - 1) return 0;
	return mapvalues[y][x];

}


