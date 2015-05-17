// This program uses the GLFW library to visualize sorting algorithms' inner works.
// necessary libraries: glfw3, opengl32
// The default sorting algorithm is std::sort.
// Algorithms implemented in C, will not be correctly displayed (they don't use class functions, for example: qsort)
// 

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <iostream>
#include <utility>
#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <climits>
#include <random>




///////////
// size of the array
size_t arraySize = 100;

// Time the thread will sleep after one swap in milliseconds
int sleep = 50;
///////////

// forward declaration
void onDisplay();

// Since two different threads work, we have to tell the drawing thread, whether it can actually draw. (otherwise it would be impossible to see which elemnts changed)
bool drawNow = false;

class Noisy {
public:
	// maximum value, which is needed by the draw function.
	static int max;
	// the two swapped values at assignment operator should be colored differently than the others.
	static size_t justChangedOne, justChangedOther;

	// number of operations the algorithm uses
	static bool began;
	static unsigned int numberofConstructor;
	static unsigned int numberofCopy;
	static unsigned int numberofDestructor;
	static unsigned int numberofMove;
	static unsigned int numberofAssignment;
	static unsigned int numberofCompare;

	//actual value, and the id
	int value;
	size_t id;

	Noisy(int val = 0, size_t id = -1) : value(val), id(id) {
		if (max < value)max = val;
		if (began) ++numberofConstructor;
	}

	~Noisy() {
		if (began)++numberofDestructor;
	}
	Noisy(Noisy const &the_other) {
		value = the_other.value;
		id = the_other.id;

		if (began)++numberofCopy;
	}

	Noisy(Noisy &&the_other){
		std::swap(value, the_other.value);
		std::swap(id, the_other.id);

		if (began)++numberofMove;
	}

	Noisy& operator=(Noisy const& the_other){
		value = the_other.value;
		justChangedOne = id;
		justChangedOther = the_other.value;

		if (began) ++numberofAssignment;

		drawNow = false;
		std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
		drawNow = true;
		return *this;
	}

	bool operator < (Noisy const&the_other){
		if (began) ++numberofCompare;

		drawNow = false;
		std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
		drawNow = true;

		return (value < the_other.value);
	}

};

//Wrapper for std sort, so it can be used by pickFunction
template <typename T>
struct stdSort{
	void operator() (std::vector<T>& vector, size_t size)
	{
		std::sort(vector.begin(), vector.end());
	}
};


//initialize static values of Noisy class
int Noisy::max = INT_MIN;
size_t Noisy::justChangedOne = -1;
size_t Noisy::justChangedOther = -1;

bool Noisy::began = false;
unsigned int Noisy::numberofConstructor = 0;
unsigned int Noisy::numberofCopy = 0;
unsigned int Noisy::numberofDestructor = 0;
unsigned int Noisy::numberofMove = 0;
unsigned int Noisy::numberofAssignment = 0;
unsigned int Noisy::numberofCompare = 0;

//array in which the elements will be stored
std::vector<Noisy> vec;

//draws the rectangle representing the value of Noisy object. if justChanged is true, it means it's been recently changed, so it should be drawn red.
void glDrawRectangle(float left, float right, float height, bool justChanged)
{
	if (justChanged) glColor3f(1.0f, 0.0f, 0.0f);
	else 
		glColor3f(0.0f, 0.0f, 1.0f);

	glBegin(GL_TRIANGLES);
	glVertex2f(left, -1.0f);
	glVertex2f(right, -1.0f);
	glVertex2f(left, height);
	glVertex2f(left, height);
	glVertex2f(right, height);
	glVertex2f(right, -1.0f);
	glEnd();

}

//iterates through the array and draws them
void drawArrayValues()
{
	float left;
	float right;
	float height;
	int i = 0;

	for (auto element : vec)
	{
		height = (static_cast<float>(element.value) / static_cast<float>(Noisy::max))*2.0f;
		left = (2.0f / static_cast<float>(vec.size())) * i;
		right = (2.0f / static_cast<float>(vec.size())) * (i + 1);

		// -1-1 intervallumon rajzoljon

		glDrawRectangle(left - 1.0f, right - 1.0f, height - 1.0f, (element.id == Noisy::justChangedOne || element.id == Noisy::justChangedOther));
		++i;
	}

	//Sometimes the algorithm would only use 1 element from the array, and use one that is not actually there. So in those cases only one should be showed as red.
	Noisy::justChangedOne = -1;
	Noisy::justChangedOther = -1;
}

//if we had several sorting functions, this would help to try out all of them
void pickFunction(std::function<void(std::vector<Noisy>&, size_t)> func) {
	func(vec, vec.size());
}

void startSort()
{
	Noisy::began = true;

	pickFunction(stdSort<Noisy>());

	//itt kicsit felelmetes a vegeredmeny, de ketseg kivul hatekony :D
	/*qsort(&vec[0], vec.size(), sizeof(Noisy), [](const void* p1, const void* p2) {

		drawNow = false;
		std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
		drawNow = true;

		return static_cast<const Noisy*>(p1)->value - static_cast<const Noisy*>(p2)->value; });*/


	std::cout << "The sorting has ended. Size of the array: " << arraySize << '\n' <<
		"number of constructors called: " << Noisy::numberofConstructor << '\n' <<
		"number of copy constructors called: " << Noisy::numberofCopy << '\n' <<
		"number of destructors called: " << Noisy::numberofDestructor << '\n' <<
		"number of assignment operators called: " << Noisy::numberofAssignment << '\n' <<
		"number of comparisons: " << Noisy::numberofCompare << '\n';
}

void Initialization() {
	//allocate vector
	vec.reserve(arraySize);

	//random generator
	std::random_device rd;
	//uniform distribution
	std::uniform_int_distribution<int> dist(1, arraySize+1);

	for (size_t i = 0; i < arraySize; ++i)
	{
		//fill vector with random values
		vec.push_back(Noisy{ dist(rd), i });
	}
}

static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

}



int main(void)
{
	Initialization();

	GLFWwindow* window;
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		exit(EXIT_FAILURE);
	window = glfwCreateWindow(600, 600, "masodik hazi", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glfwSetKeyCallback(window, key_callback);

	//start thread that will sort the array
	std::thread t{ startSort };

	t.detach();

	while (!glfwWindowShouldClose(window))
	{
		if (drawNow)
		{
			float ratio;
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			ratio = width / (float)height;
			glViewport(0, 0, width, height);
			glClear(GL_COLOR_BUFFER_BIT);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			if (Noisy::began){
				drawArrayValues();
			}

			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
