//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2014-tol.          
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk. 
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat. 
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni (printf is fajlmuvelet!)
// - new operatort hivni az onInitialization függvényt kivéve, a lefoglalt adat korrekt felszabadítása nélkül 
// - felesleges programsorokat a beadott programban hagyni
// - tovabbi kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan gl/glu/glut fuggvenyek hasznalhatok, amelyek
// 1. Az oran a feladatkiadasig elhangzottak ES (logikai AND muvelet)
// 2. Az alabbi listaban szerepelnek:  
// Rendering pass: glBegin, glVertex[2|3]f, glColor3f, glNormal3f, glTexCoord2f, glEnd, glDrawPixels
// Transzformaciok: glViewport, glMatrixMode, glLoadIdentity, glMultMatrixf, gluOrtho2D, 
// glTranslatef, glRotatef, glScalef, gluLookAt, gluPerspective, glPushMatrix, glPopMatrix,
// Illuminacio: glMaterialfv, glMaterialfv, glMaterialf, glLightfv
// Texturazas: glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glTexEnvi, 
// Pipeline vezerles: glShadeModel, glEnable/Disable a kovetkezokre:
// GL_LIGHTING, GL_NORMALIZE, GL_DEPTH_TEST, GL_CULL_FACE, GL_TEXTURE_2D, GL_BLEND, GL_LIGHT[0..7]
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Srajner Ferenc
// Neptun : YX3X5I
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy 
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem. 
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a 
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb 
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem, 
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.  
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat 
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <iostream>

#if defined(__APPLE__)                                                                                                                                                                                                            
#include <OpenGL/gl.h>                                                                                                                                                                                                            
#include <OpenGL/glu.h>                                                                                                                                                                                                           
#include <GLUT/glut.h>                                                                                                                                                                                                            
#else                                                                                                                                                                                                                             
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)                                                                                                                                                                       
#include <windows.h>                                                                                                                                                                                                              
#endif                                                                                                                                                                                                                            
#include <GL/gl.h>                                                                                                                                                                                                                
#include <GL/glu.h>                                                                                                                                                                                                               
#include <GL/glut.h>                                                                                                                                                                                                              
#endif          


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...


struct Vector {
	float x, y, z;

	Vector() {
		x = y = z = 0;
	}
	Vector(float x0, float y0, float z0 = 0.0) {
		x = x0; y = y0; z = z0;
	}
	Vector(const Vector& asd){
		x = asd.x;
		y = asd.y;
		z = asd.z;
	}
	Vector operator*(float a) {
		return Vector(x * a, y * a, z * a);
	}
	Vector operator/(float a) {
		return Vector(x / a, y / a, z / a);
	}
	Vector operator+(const Vector& v) {
		return Vector(x + v.x, y + v.y, z + v.z);
	}
	Vector operator+=(const Vector& v) {
		return Vector(x += v.x, y += v.y, z += v.z);
	}
	Vector operator-(const Vector& v) {
		return Vector(x - v.x, y - v.y, z - v.z);
	}
	float operator*(const Vector& v) {
		return (x * v.x + y * v.y + z * v.z);
	}
	Vector operator%(const Vector& v) {
		return Vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}
	float Length() { return sqrt(x * x + y * y + z * z); }

	float Distance(Vector a)
	{
		return sqrt((x - a.x)*(x - a.x) + (y - a.y)*(y - a.y) + (z- a.z)*(z - a.z));
	}
};


struct Color {
	float r, g, b;

	Color() {
		r = g = b = 0;
	}
	Color(float r0, float g0, float b0) {
		r = r0; g = g0; b = b0;
	}
	Color operator*(float a) {
		return Color(r * a, g * a, b * a);
	}
	Color operator*(const Color& c) {
		return Color(r * c.r, g * c.g, b * c.b);
	}
	Color operator+(const Color& c) {
		return Color(r + c.r, g + c.g, b + c.b);
	}
};



struct ControllPoint
{
	Vector Point;
	int time;
	int StartTime;
	Vector Center;
	

	ControllPoint()
	{
		Point.x = 0;
		Point.y = 0;
	}
	
	ControllPoint(float x0, float y0, int t0)
	{
		Point.x = x0;
		Point.y = y0;
		time = t0;
		StartTime = t0;
		Center.x = x0;
		Center.y = y0 + 5.0f;
	}

};


ControllPoint ControllPoints[10];
int counter = 0;


int prevtime;
bool animation = false;
float animradius = 5.0f;


const int screenWidth = 600;
const int screenHeight = 600;
int width = 68;
int height = 58;


float startwidth = 0;
float startheight = 0;

float* centerx=&startwidth;
float* centery=&startheight;


//--------------------------------------------------------
// Convex Hull:
// Monotone chain algorithm (Andrew's algorithm)
// source: http://geomalgorithms.com/a10-_hull-1.html
//--------------------------------------------------------
struct ConvexHull
{
	static void Draw()
	{
		ControllPoint* ConvexH;
		int convexcount;
		int *sizepointer = &convexcount;
		convexcount = counter;
		ConvexH = Calculate(ControllPoints, sizepointer);
		

	
		glColor3f(0.25098f, 0.8784313f, 0.81568627f);
		glBegin(GL_TRIANGLE_FAN);
		for (int i = 0; i < convexcount; i++) {
			glVertex2f(ConvexH[i].Point.x, ConvexH[i].Point.y);
		}
		glEnd();
		delete[] ConvexH;

	}
	static void sortArray(ControllPoint Points[], int n)
	{
		bool swapped = true;
		int j = 0;
		ControllPoint tmp;
		while (swapped) {
			swapped = false;
			j++;
			for (int i = 0; i < n - j; i++) {
				if (Points[i].Point.x > Points[i + 1].Point.x || (Points[i].Point.x == Points[i + 1].Point.x && Points[i].Point.y > Points[i + 1].Point.y)) {
					tmp = Points[i];
					Points[i] = Points[i + 1];
					Points[i + 1] = tmp;
					swapped = true;
				}
			}
		}
	}

	static float cross(const ControllPoint &O, const ControllPoint &A, const ControllPoint &B)
	{
		return (A.Point.x - O.Point.x) * (B.Point.y - O.Point.y) - (A.Point.y - O.Point.y) * (B.Point.x - O.Point.x);
	}


	static ControllPoint* Calculate(ControllPoint* Points, int *size){


		int k = 0;

		ControllPoint* temp = new ControllPoint[2 * (*size)];


		ControllPoint* tmp = new ControllPoint[10];
		for (int a = 0; a < counter; a++) tmp[a] = ControllPoints[a];

		sortArray(tmp, *size);


		for (int i = 0; i < *size; ++i) {
			while (k >= 2 && cross(temp[k - 2], temp[k - 1], tmp[i]) <= 0) k--;
			temp[k++] = tmp[i];
		}

		for (int i = *size - 2, t = k + 1; i >= 0; i--) {
			while (k >= t && cross(temp[k - 2], temp[k - 1], tmp[i]) <= 0) k--;
			temp[k++] = tmp[i];
		}
		delete[] tmp;
		*size = k;
		return temp;
	}
};


//--------------------------------------------------------
// Bezier curve
// source: előadásdiák
//--------------------------------------------------------
struct Bezier
{

	static void Draw()

	{
		glColor3f(1.0f, 0.0f, 0.0f);
		glPointSize(1.35f);
		glBegin(GL_LINES);
		Vector x;
		Vector y=ControllPoints[0].Point;
		for (float t = (1.0 / (100 * counter)); t < 1.0f; t += (1.0 / (100 * counter))){
			x = CalculateBezierPoint(t);
			glVertex2f(x.x, x.y);
			glVertex2f(y.x, y.y);
			y = x;
		}
		glEnd();
	}

	static Vector CalculateBezierPoint(float t) {
		ControllPoint* tmp = new ControllPoint[10];
		for (int a = 0; a < counter; a++) tmp[a] = ControllPoints[a];
		int i = counter - 1;
		while (i > 0) {
			for (int k = 0; k < i; k++)
				tmp[k].Point = tmp[k].Point + ((tmp[k + 1].Point - tmp[k].Point)*t);
			i--;
		}
		Vector answer = tmp[0].Point;
		delete[] tmp;
		return answer;
	}
};

//--------------------------------------------------------
// Catmull-Rom curve
// source: előadásdiák
//--------------------------------------------------------
struct CatmullRom
{
	static void Draw()
	{

		glColor3f(0.0f, 1.0f, 0.0f);
		glBegin(GL_POINTS);
		Vector vtmp;
		for (int i = 0; i < counter - 1; i++)
		{
			Vector a0 = ControllPoints[i].Point;
			Vector a1 = getVi(i);
			Vector a2 = ((((ControllPoints[i + 1].Point - ControllPoints[i].Point) * 3.0)* (1.0 / pow((ControllPoints[i + 1].time - ControllPoints[i].time), 2.0))) -
				((getVi(i + 1) + (getVi(i) * 2.0))* (1.0 / (ControllPoints[i + 1].time - ControllPoints[i].time))));

			Vector a3 = ((((ControllPoints[i].Point - ControllPoints[i + 1].Point) * 2.0)* (1.0 / pow((ControllPoints[i + 1].time - ControllPoints[i].time), 3.0))) +
				((getVi(i + 1) + getVi(i))* (1.0 / pow((ControllPoints[i + 1].time - ControllPoints[i].time), 2.0))));

			for (int j = ControllPoints[i].time; j < ControllPoints[i + 1].time; j += 1.0f)
			{
				vtmp = ((a3*pow(j - ControllPoints[i].time, 3.0)) + (a2*pow(j - ControllPoints[i].time, 2.0)) + (a1*(j - ControllPoints[i].time)) + a0);
				glVertex2f(vtmp.x, vtmp.y);
			}

		}
		glEnd();
	}


	static Vector getVi(int i)
	{
		if (i == 0 || i == counter - 1)
			return Vector(0.0, 0.0);

		return ((((ControllPoints[i + 1].Point - ControllPoints[i].Point) / (ControllPoints[i + 1].time - ControllPoints[i].time)) +
			((ControllPoints[i].Point - ControllPoints[i - 1].Point) / (ControllPoints[i].time - ControllPoints[i - 1].time)))*0.5);
	}
};


//--------------------------------------------------------
// Catmull-Clark curve
// source: előadásdiák
//--------------------------------------------------------
struct CatmullClark
{
	static void Draw()
	{
		ControllPoint*	CatmullClark;
		int CatmullSize=counter;
		int catmullIterations=5;
		
		for (int i = 0; i < catmullIterations; i++)
			CatmullSize = ((2 * CatmullSize) - 1);


		glPointSize(3.0f);
		glColor3f(0.0f, 0.0f, 1.0f);
		glBegin(GL_LINES);
		
		CatmullClark = Calculate(ControllPoints, counter, catmullIterations);
		
		Vector asd;
		Vector dsa;

		for (int i = 1; i < CatmullSize; i++)
		{
			asd = CatmullClark[i - 1].Point;
			dsa = CatmullClark[i].Point;
			glVertex2f(asd.x, asd.y);
			glVertex2f(dsa.x, dsa.y);

		}
		glEnd();
		delete[] CatmullClark;
	}

	static ControllPoint* Iterate(ControllPoint* Points, int size)
	{
		
		ControllPoint* retPoints = new ControllPoint[(2 * (size)) - 1];


		for (int i = 0; i < size-1; i++)
		{
			if (i == 0)
			{
				retPoints[i] = Points[0];
				retPoints[(2 * size) - 2] = Points[size-1];
			}
			else
				retPoints[2 * i].Point = ((Points[i].Point*0.5f) + (((Points[i - 1].Point + Points[i].Point)*0.5f)*0.25f) + (((Points[i].Point + Points[i+1].Point)*0.5f)*0.25f));

			retPoints[(2 * i) + 1].Point = ((Points[i + 1].Point + Points[i].Point)*0.5f);
}
		
		return retPoints;
	}
	static ControllPoint* Calculate (ControllPoint* Points, int size, int iterations)
	{
		int CatmullSize=size;
		int itersize = size;
		for (int i = 0; i < iterations; i++)
			CatmullSize = ((2 * CatmullSize)-1);

			ControllPoint* returnArray = new ControllPoint[CatmullSize];
			
			for (int i = 0; i < itersize; i++) returnArray[i] = Points[i];

			ControllPoint* returnValues;

			for (int i = 0; i < iterations; i++)
			{
				returnValues = Iterate(returnArray, itersize);
				itersize = 2 * itersize - 1;
				for (int i = 0; i < itersize; i++) returnArray[i] = returnValues[i];
				delete[] returnValues;
			}



		return returnArray;
	}
};





void onInitialization() {
	glViewport(0, 0, screenWidth, screenHeight);

	
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);		
	

}


void onDisplay() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	gluOrtho2D(*centerx - 0.5*width, *centerx + 0.5*width, *centery + 0.5*height, *centery - 0.5*height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	



	
	

	glColor3f(1.0f, 1.0f, 1.0f);


	
	if (counter > 1)
	{
		ConvexHull::Draw();

		Bezier::Draw();

		CatmullRom::Draw();

		CatmullClark::Draw();
		
	}


	glColor3f(0.0f, 0.0f, 0.0f);
	float radius = 2.0f;
	for (int i = 0; i < counter; i++){
		glBegin(GL_TRIANGLE_FAN);
		glVertex2f(ControllPoints[i].Point.x, ControllPoints[i].Point.y);
		for (int j = 0; j <= 360; j++) {


			float angle = float(j) / 360 * 2.0f * M_PI;
			glVertex2f(ControllPoints[i].Point.x + radius*cos(angle), ControllPoints[i].Point.y + radius*sin(angle));
		}

		glEnd();
	}



	glColor3f(1.0f, 1.0f, 1.0f);
	glPointSize(3.0f);
	glBegin(GL_POINTS);
	for (int i = 0; i < counter; i++){
		glVertex2f(ControllPoints[i].Point.x, ControllPoints[i].Point.y);
	}
	glEnd();



	if (animation)
	{
		int dT;
		float angle;
		for (int i = 0; i < counter; i++)
		{
			dT = glutGet(GLUT_ELAPSED_TIME) - ControllPoints[i].StartTime;

			float j = dT / (5000 / 360);
			if (i % 2 == 0)
				angle = j / 360 * 2.0f * M_PI;

			else angle = -1 * j / 360 * 2.0f*M_PI;

			angle += M_PI / 2;
			ControllPoints[i].Point.x = ControllPoints[i].Center.x - animradius*cos(angle);
			ControllPoints[i].Point.y = ControllPoints[i].Center.y - animradius*sin(angle);
		}

	}



	prevtime = glutGet(GLUT_ELAPSED_TIME);
	
	glutSwapBuffers();     				
}


void onKeyboard(unsigned char key, int x, int y) {
	if (key == 'd') glutPostRedisplay(); 		

	if (key == ' ' && animation == false)
	{
		for (int i = 0; i < counter; i++)
			ControllPoints[i].StartTime = glutGet(GLUT_ELAPSED_TIME);
			animation = true;
	}

}


void onKeyboardUp(unsigned char key, int x, int y) 
{
	

}


void onMouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)  
	{

		if (counter < 10){
			float a = ((float)x / screenWidth)  * width + *centerx-width/2;
			float b = ((float)y / screenHeight) * height + *centery-height/2;

			ControllPoint *point = new ControllPoint(a, b, glutGet(GLUT_ELAPSED_TIME));

			ControllPoints[counter] = *point;

			counter++;
		}
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		bool found = false;
		Vector a(((float)x / screenWidth)  * width + *centerx - width / 2, ((float)y / screenHeight) * height + *centery - height / 2);
		for (int i = 0; i < counter; i++)
		{

			float dis = ControllPoints[i].Point.Distance(a);

			if (ControllPoints[i].Point.Distance(a) <= 2.0)
			{
				centerx = &(ControllPoints[i].Point.x);
				centery = &(ControllPoints[i].Point.y);
				found = true;
			}
		
		}
		if (!found)
		{
			centerx = &startwidth;
			centery = &startheight;
		}
	}

		glutPostRedisplay(); 						 
}


void onMouseMotion(int x, int y)
{

}


void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);		

	if (animation)
	{
		glutPostRedisplay();
	}


	
}

// ...Idaig modosithatod
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A C++ program belepesi pontja, a main fuggvenyt mar nem szabad bantani
int main(int argc, char **argv) {
	glutInit(&argc, argv); 				// GLUT inicializalasa
	glutInitWindowSize(600, 600);			// Alkalmazas ablak kezdeti merete 600x600 pixel 
	glutInitWindowPosition(100, 100);			// Az elozo alkalmazas ablakhoz kepest hol tunik fel
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);	// 8 bites R,G,B,A + dupla buffer + melyseg buffer

	glutCreateWindow("Grafika hazi feladat");		// Alkalmazas ablak megszuletik es megjelenik a kepernyon

	glMatrixMode(GL_MODELVIEW);				// A MODELVIEW transzformaciot egysegmatrixra inicializaljuk
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);			// A PROJECTION transzformaciot egysegmatrixra inicializaljuk
	glLoadIdentity();

	onInitialization();					// Az altalad irt inicializalast lefuttatjuk

	glutDisplayFunc(onDisplay);				// Esemenykezelok regisztralasa
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();					// Esemenykezelo hurok

	return 0;
}