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

float ToRadian(float source)
{
	return (M_PI / 180.f) * source;
}

float ToDegree(float source)
{
	return (180.f / M_PI) * source;
}

float random(float min, float max)
{
	float ret = (float)(rand() / RAND_MAX)*max + min;
	if (ret > max) return ret - max + min;
	return ret;
}

//--------------------------------------------------------
// 3D Vektor
//--------------------------------------------------------
struct Vector {
	float x, y, z;

	Vector() {
		x = y = z = 0;
	}
	Vector(float x0, float y0, float z0 = 0) {
		x = x0; y = y0; z = z0;
	}
	Vector operator*(const float a) const {
		return Vector(x * a, y * a, z * a);
	}
	Vector operator/(const float a) const {
		return Vector(x / a, y / a, z / a);
	}
	Vector operator+(const Vector& v) const {
		return Vector(x + v.x, y + v.y, z + v.z);
	}
	Vector operator-(const Vector& v) const {
		return Vector(x - v.x, y - v.y, z - v.z);
	}
	Vector operator/(const Vector& v) const {
		return Vector(x / v.x, y / v.y, z / v.z);
	}

	float operator*(const Vector& v) const {
		return (x * v.x + y * v.y + z * v.z);
	}

	Vector operator%(const Vector& v) {
		return Vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}
	Vector operator^(const Vector& v) const {
		return Vector(x * v.x, y * v.y, z * v.z);
	}

	float Length() { return sqrt(x * x + y * y + z * z); }

	float Distance(const Vector& v){ return sqrt((x - v.x)*(x - v.x) + (y - v.y)*(y - v.y) + (z - v.z)*(z - v.z)); }

	Vector& normalize()
	{
		float length = Length();
		if (length > 0) {
			float normal = 1.0f / length;

			x *= normal, y *= normal, z *= normal;
		}
		return *this;
	}

};

//--------------------------------------------------------
// Spektrum illetve szin
//--------------------------------------------------------
struct Color {
	float r, g, b;

	Color() {
		r = g = b = 0;
	}
	Color(float r0, float g0, float b0) {
		r = r0; g = g0; b = b0;
	}
	Color(float all) {
		r = all; g = all; b = all;
	}
	Color operator*(float a) {
		return Color(r * a, g * a, b * a);
	}
	Color operator*(const Color& c) const {
		return Color(r * c.r, g * c.g, b * c.b);
	}
	Color operator+(const Color& c)  const {
		return Color(r + c.r, g + c.g, b + c.b);
	}
	Color operator-(float a) {
		return Color(r - a, g - a, b - a);
	}
	Color operator-(const Color& c) {
		return Color(r - c.r, g - c.g, b - c.b);
	}
	Color operator/(const Color& c) {
		return Color(r / c.r, g / c.g, b / c.b);
	}

	Color operator += (const Color &v) { r += v.r, g += v.g, b += v.b; return *this; }

};

//--------------------------------------------------------
// Triangle
//--------------------------------------------------------
void glVertexV(const Vector& v) {

	glVertex3f(v.x, v.y, v.z);
}

void uv(const Vector& Point)
{
	float u = 0.5f + atan2(Point.x, Point.z) / (2.0f * M_PI);
	float v = 0.5f - asin(Point.y) / M_PI;
	glTexCoord2f(u, v);
}

void DrawTriangle(const Vector& a, const Vector& b, const Vector& c)
{
	Vector normal = ((a - b) % (a - c)).normalize();

	glNormal3f(normal.x, normal.y, normal.z);
	glVertexV(c);
	glVertexV(b);
	glVertexV(a);
}


//--------------------------------------------------------
// Globals
//--------------------------------------------------------
const int screenWidth = 600;
const int screenHeight = 600;

long prevTime=0;
//--------------------------------------------------------
// Textures
//--------------------------------------------------------
const int SizeofPlanet = 256;
const int SizeofSB = 384;
const int SizeofSun = 256;
GLubyte PlanetP[SizeofPlanet][SizeofPlanet][3];
GLubyte SpaceBoxP[SizeofSB][SizeofSB][3];
GLubyte SunP[SizeofSun][SizeofSun][3];

//--------------------------------------------------------
// Colours
// http://www.nicoptere.net/dump/materials.html
//--------------------------------------------------------
float LightDiff[] = { 0.5f, 0.5f, 0.5f, 1.0f };
float LightSpec[] = { 0.2f, 0.2f, 0.2f, 1.0f };
float LightShine[] = { 180 };

const float panelDiff[] = { 0.53464, 0.5345, 0.776534, 1.0f };
const float panelSpec[] = { 0.46578, 0.496, 0.72632, 1.0f };
const float panelShine[] = { 102.0f };

const float silverDiff[] = { 0.50754f, 0.50754f, 0.50754f, 1.0f };
const float silverSpec[] = { 0.508273f, 0.508273f, 0.508273f, 1.0f };
const float silverShine[] = { 102.0f };

const float chromeDiff[] = { 0.4f, 0.4f, 0.4f, 1.0f };
const float chromeSpec[] = { 0.774597f, 0.774597f, 0.774597, 1.0f };
const float chromeShine[] = { 76.8f };

const float copperDiff[] = { 0.7038f, 0.27048f, 0.0828f, 1.0f };
const float copperSpec[] = { 0.256777f, 0.137622f, 0.086014f, 1.0f };
const float copperShine[] = { 25.0f };

const float turquoiseDiff[] = { 0.396f, 0.74151f, 0.69102f, 1.0f };
const float turquoiseSpec[] = { 0.297254f, 0.30829f, 0.306678f, 1.0f };
const float turquoiseShine[] = { 12.8f };

const float jadeDiff[] = { 0.54f, 0.89f, 0.63f, 1.0f };
const float jadeSpec[] = { 0.316228f, 0.316228f, 0.316228f, 1.0f };
const float jadeShine[] = { 12.8f };

const float pearlDiff[] = { 1.0f, 0.829f, 0.829f, 1.0f };
const float pearlSpec[] = { 0.296648f, 0.296648f, 0.296648f, 1.0f };
const float pearlShine[] = { 11.264f };

const float air[] = { 0.4f, 0.6f, 1.0f, 0.35f };

float bilboardcolour[] = { 0.0f, 0.0f, 0.0f, 0.2f };
const float bilboardblue[] = { 0.1f, 0.7f, 0.68f, 0.5f };
const float bilboardyellow[] = { 1.0f, 1.0f, 0.0f, 0.5f };
const float bilboardred[] = { 1.0f, 0.55f, 0.0f, 0.5f };



const float black[] = { 0, 0, 0, 1 };
const float white[] = { 1, 1, 1, 1 };

//--------------------------------------------------------
// Perlin
//--------------------------------------------------------
struct Perlin
{
	static float InterPolation(float a, float b, float c)
	{
		return a + (b - a)*c*c*(3 - 2 * c);
	}

	static float InterLinear(float a, float b, float c)
	{
		return a*(1 - c) + b*c;
	}

	static float Noise(int x)
	{
		return (((x * (x * x * 15731 + 789221))) / 5147483648.0);
	}

	static float PerlinNoise(float x, float y, int width, int octaves, int seed, double periode)
	{
		double a, b, value, freq, zone_x, zone_y;
		int s, box, num, step_x, step_y;
		int amplitude = 120;
		int noisedata;

		freq = 1 / (float)(periode);

		for (s = 0; s<octaves; s++)
		{
			num = (int)(width*freq);
			step_x = (int)(x*freq);
			step_y = (int)(y*freq);
			zone_x = x*freq - step_x;
			zone_y = y*freq - step_y;
			box = step_x + step_y*num;
			noisedata = (box + seed);
			a = InterPolation(Noise(noisedata), Noise(noisedata + 1), zone_x);
			b = InterPolation(Noise(noisedata + num), Noise(noisedata + 1 + num), zone_x);

			value = InterPolation(a, b, zone_y)*amplitude;
		}
		return value;
	}

	static float scramble(int x, int y)
	{

		int seed;
		int width;
		float  disp1, disp2, disp3, disp4, disp5, disp6, scale;


		scale = 10;
		width = 12325;
		seed = 10;

		disp1 = PerlinNoise(x*scale, y*scale, width, 1, seed, 100);
		disp2 = PerlinNoise(x*scale, y*scale, width, 2, seed, 75);
		disp3 = PerlinNoise(x*scale, y*scale, width, 3, seed, 12.5);
		disp4 = PerlinNoise(x*scale, y*scale, width, 4, seed, 6.25);
		disp5 = PerlinNoise(x*scale, y*scale, width, 5, seed, 3.125);
		disp6 = PerlinNoise(x*scale, y*scale, width, 6, seed, 1.56);

		float valor = (disp1)+(disp2*0.5) + (disp3*0.25) + (disp4*0.125) + (disp5*0.03125) + (disp6*0.0156);


		return valor;
	}

};


//--------------------------------------------------------
// Material
//--------------------------------------------------------
struct Material
{
	unsigned int texture_id;
	GLint internalformat;
	int height, width;
	GLenum format;
	GLvoid* pixels;

	void Bind(GLint internalformat, int width, int height, GLenum format, const GLvoid* pixels) {
		glGenTextures(1, &texture_id);
		glBindTexture(GL_TEXTURE_2D, texture_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, internalformat, width, height, 0, format, GL_UNSIGNED_BYTE, pixels);
	}
};

//--------------------------------------------------------
// Object
//--------------------------------------------------------
struct Object
{
	Vector Position, Rotation, Scale;
	float angle;
	Material texture;

	virtual void Draw() = 0;
};



//--------------------------------------------------------
// Plane
//--------------------------------------------------------
struct Plane : public Object
{

	Plane(const Vector& pos = Vector(0, 0, 0), const Vector& rot = Vector(0, 0, 0), const Vector& scale = Vector(1, 1, 1), const float ag = 0)
	{
		Position = pos;
		Rotation = rot;
		Scale = scale;
		angle = ag;
	}

	void Draw()
	{
		glPushMatrix();

		

		glDisable(GL_CULL_FACE);
		glTranslatef(Position.x, Position.y, Position.z);
		glRotatef(angle, Rotation.x, Rotation.y, Rotation.z);
		glScalef(Scale.x, Scale.y, Scale.z);

		glBegin(GL_TRIANGLES);
		DrawTriangle(Vector(-0.5, 0.5, 0.0f), Vector(-0.5, -0.5, 0.0f), Vector(0.5, 0.5, 0.0f));
		DrawTriangle(Vector(-0.5, -0.5, 0.0f), Vector(0.5, -0.5, 0.0f), Vector(0.5, 0.5, 0.0f));
		glEnd();
		glEnable(GL_CULL_FACE);
		glPopMatrix();

	}
};

//--------------------------------------------------------
// Sphere
//http://hu.wikipedia.org/wiki/Ikozaéder
//--------------------------------------------------------
struct Sphere : public Object
{
	static const Vector Points[12];
	static const Vector TriangleIndices[20];


	int depth;
	bool textured;
	bool transparent;
	long spin;

	Sphere(const Vector& pos = Vector(0, 0, 0), const Vector& rot = Vector(0, 0, 0), const Vector& scale = Vector(1, 1, 1), const float ag = 0, const int d = 3)
	{
		Position = pos;
		Rotation = rot;
		Scale = scale;
		angle = ag;
		depth = d;
		textured = false;
		transparent = false;

		spin = 0;
	}


	void TexturedTriangle(const Vector& a, const Vector& b, const Vector& c)
	{
		Vector normal = ((a - b) % (a - c)).normalize();
		glNormal3f(normal.x, normal.y, normal.z);

		uv(c);
		glVertex3f(c.x, c.y, c.z);

		uv(b);
		glVertex3f(b.x, b.y, b.z);

		uv(a);
		glVertex3f(a.x, a.y, a.z);
	}

	void DrawTriangles(Vector a, Vector b, Vector c, int depth)
	{
		if (depth <= 0)
		{
			a.normalize(); b.normalize(); c.normalize();

			if (textured) TexturedTriangle(a, b, c);
			else DrawTriangle(a, b, c);
		}
		else
		{
			Vector ab = ((a + b) / 2).normalize();
			Vector ac = ((a + c) / 2).normalize();
			Vector bc = ((b + c) / 2).normalize();

			DrawTriangles(a, ab, ac, depth - 1);
			DrawTriangles(b, bc, ab, depth - 1);
			DrawTriangles(c, ac, bc, depth - 1);
			DrawTriangles(ab, bc, ac, depth - 1);
		}
	}

	void Draw()
	{
		glPushMatrix();
		

		glTranslatef(Position.x, Position.y, Position.z);
		glRotatef(angle, Rotation.x, Rotation.y, Rotation.z);
		glScalef(Scale.x, Scale.y, Scale.z);


		if (textured)
		{
			long deltaT = glutGet(GLUT_ELAPSED_TIME) - prevTime;
			spin += deltaT*0.09;
				glRotatef(spin, 0, 1, 1);

			glEnable(GL_TEXTURE_2D);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, white);
			glMaterialfv(GL_FRONT, GL_SPECULAR, white);
			glMaterialfv(GL_FRONT, GL_SHININESS, white);
			glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
			glBindTexture(GL_TEXTURE_2D, texture.texture_id);
		}

		if (transparent)
		{
			glDisable(GL_CULL_FACE);
			glEnable(GL_BLEND);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, air);
			glMaterialfv(GL_FRONT, GL_SPECULAR, black);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}
		glBegin(GL_TRIANGLES);
		for (int i = 0; i < 20; i++)
			DrawTriangles(Points[(int)TriangleIndices[i].x], Points[(int)TriangleIndices[i].y], Points[(int)TriangleIndices[i].z], depth);
		glEnd();

		if (textured) glDisable(GL_TEXTURE_2D);
		if (transparent)
		{
			glDisable(GL_BLEND);
			glEnable(GL_CULL_FACE);
		}

		glPopMatrix();
	}

};

const Vector Sphere::Points[12] =
{
	Vector(-1, 0.0, 1), Vector(1, 0.0, 1), Vector(-1, 0.0, -1), Vector(1, 0.0, -1),
	Vector(0.0, 1, 1), Vector(0.0, 1, -1), Vector(0.0, -1, 1), Vector(0.0, -1, -1),
	Vector(1, 1, 0.0), Vector(-1, 1, 0.0), Vector(1, -1, 0.0), Vector(-1, -1, 0.0)

};

const Vector Sphere::TriangleIndices[20] =
{
	Vector(0, 4, 1), Vector(0, 9, 4), Vector(9, 5, 4), Vector(4, 5, 8), Vector(4, 8, 1),
	Vector(8, 10, 1), Vector(8, 3, 10), Vector(5, 3, 8), Vector(5, 2, 3), Vector(2, 7, 3),
	Vector(7, 10, 3), Vector(7, 6, 10), Vector(7, 11, 6), Vector(11, 0, 6), Vector(0, 1, 6),
	Vector(6, 1, 10), Vector(9, 0, 11), Vector(9, 11, 2), Vector(9, 2, 5), Vector(7, 2, 11)
};




//--------------------------------------------------------
// CatmullRomObject
// source: önerő
//--------------------------------------------------------
struct CatmullRomObject : public Object
{
	Vector *ControllPoints;
	int counter;
	int steepness;


	CatmullRomObject(int size = 12)
	{
		steepness = 500;
		counter = size;
		ControllPoints = new Vector[counter];

		ControllPoints[0].x = 0.1f;
		ControllPoints[0].y = 1.0f;

		ControllPoints[1].x = 0.17f;
		ControllPoints[1].y = 0.96f;

		ControllPoints[2].x = 0.19f;
		ControllPoints[2].y = 0.78f;

		ControllPoints[3].x = 0.08f;
		ControllPoints[3].y = 0.65;

		ControllPoints[4].x = 0.22f;
		ControllPoints[4].y = 0.53f;

		ControllPoints[5].x = 0.21f;
		ControllPoints[5].y = 0.45f;

		ControllPoints[6].x = 0.215f;
		ControllPoints[6].y = 0.4f;

		ControllPoints[7].x = 0.15f;
		ControllPoints[7].y = 0.38f;

		ControllPoints[8].x = 0.218f;
		ControllPoints[8].y = 0.36f;

		ControllPoints[9].x = 0.18f;
		ControllPoints[9].y = 0.2f;

		ControllPoints[10].x = 0.12f;
		ControllPoints[10].y = 0.1f;

		ControllPoints[11].x = 0.0f;
		ControllPoints[11].y = 0.0f;
	}

	void Draw()
	{

		glPushMatrix();
		Vector top = GetPoint(0);
		Vector bottom;
		for (double j = (1.0f / 100.0f); j < 0.91; j += (1.0f / 100.0f))
		{
			bottom = GetPoint(j);

			for (int i = 0; i < 360; i += 15)
			{
				float angle = ToRadian(i);
				float nextangle = ToRadian(i + 15);

				//z nem erdekel
				//xl    xr
				//yl    yr
				Vector xl(top.x*cos(angle), top.y, top.x*sin(angle));
				Vector yl(bottom.x*cos(angle), bottom.y, bottom.x*sin(angle));
				Vector xr(top.x*cos(nextangle), top.y, top.x*sin(nextangle));
				Vector yr(bottom.x*cos(nextangle), bottom.y, bottom.x*sin(nextangle));

				glBegin(GL_TRIANGLES);
				DrawTriangle(xl, yl, xr);
				DrawTriangle(yl, yr, xr);
				glEnd();

			}
			top = bottom;
		}
		glPopMatrix();
	}
	Vector GetPoint(double interpolation)
	{
		int i = (int)((counter - 1)*interpolation);

		Vector vtmp;

		Vector a0 = ControllPoints[i];
		Vector a1 = getVi(i);
		Vector a2 = ((((ControllPoints[i + 1] - ControllPoints[i]) * 3.0)* (1.0 / pow(steepness, 2.0))) -
			((getVi(i + 1) + (getVi(i) * 2.0))* (1.0 / steepness)));



		Vector a3 = ((((ControllPoints[i] - ControllPoints[i + 1]) * 2.0)* (1.0 / pow(steepness, 3.0))) +
			((getVi(i + 1) + getVi(i))* (1.0 / pow(steepness, 2.0))));




		vtmp = ((a3*pow((counter*steepness*interpolation - i*steepness), 3)) + (a2*pow((counter*steepness*interpolation - i*steepness), 2)) + (a1*(counter*steepness*interpolation - i*steepness)) + a0);
		return vtmp;
	}

	Vector getVi(int i)
	{
		if (i == 0 || i == counter - 1)
			return Vector(0.0, 0.0);

		return ((((ControllPoints[i + 1] - ControllPoints[i]) / steepness) +
			((ControllPoints[i] - ControllPoints[i - 1]) / steepness))*0.5f);
	}




};

struct Cylinder : public Object
{
	Vector *ControllPoints;
	int counter;
	bool closed;

	Cylinder(int size = 30, bool close = true)
	{
		closed = close;
		counter = size;
		ControllPoints = new Vector[counter];
		float radius = 1.0f;
		for (int i = 0; i < counter - 1; i++)
		{
			float angle = ToRadian((i + 1)*(360.0f / counter));

			ControllPoints[i].x = radius*cos(angle);
			ControllPoints[i].y = radius*sin(angle);
		}
		ControllPoints[counter - 1].x = ControllPoints[0].x;
		ControllPoints[counter - 1].y = ControllPoints[0].y;
	}

	void Draw()
	{
		glPushMatrix();

		for (int i = 0; i < counter - 1; i++)
		{
			glDisable(GL_CULL_FACE);
			glBegin(GL_TRIANGLES);
			DrawTriangle(Vector(ControllPoints[i].x, 0.5, ControllPoints[i].y), Vector(ControllPoints[i].x, -0.5, ControllPoints[i].y), Vector(ControllPoints[i + 1].x, 0.5, ControllPoints[i + 1].y));
			DrawTriangle(Vector(ControllPoints[i].x, -0.5, ControllPoints[i].y), Vector(ControllPoints[i + 1].x, -0.5, ControllPoints[i + 1].y), Vector(ControllPoints[i + 1].x, 0.5, ControllPoints[i + 1].y));
			glEnd();
			glEnable(GL_CULL_FACE);

		}

		if (closed)
		{
			glBegin(GL_TRIANGLES);
			for (int i = 1; i < counter - 3; i++)
			{
				DrawTriangle(Vector(ControllPoints[0].x, 0.5, ControllPoints[0].y), Vector(ControllPoints[i].x, 0.5, ControllPoints[i].y), Vector(ControllPoints[i + 1].x, 0.5, ControllPoints[i + 1].y));
			}
			glEnd();
			glBegin(GL_TRIANGLES);
			for (int i = 1; i < counter - 3; i++)
			{
				DrawTriangle(Vector(ControllPoints[0].x, -0.5, ControllPoints[0].y), Vector(ControllPoints[i].x, -0.5, ControllPoints[i].y), Vector(ControllPoints[i + 1].x, -0.5, ControllPoints[i + 1].y));
			}
			glEnd();
		}


		glPopMatrix();
	}

};

struct SkyBox : public Object
{

	SkyBox(const Vector& pos = Vector(0, 0, 0), const Vector& rot = Vector(0, 0, 0), const Vector& scale = Vector(30, 30, 30), float ag = 0)
	{
		Position = pos;
		Rotation = rot;
		Scale = scale;
		angle = ag;
	}


	void DrawSquare(const Vector& a, const Vector& b, const Vector& c, const Vector& d) {
		Vector normal = ((b - a) % (a - d)).normalize();
		glNormal3f(normal.x, normal.y, normal.z);
		glTexCoord2f(0, 0);
		glVertexV(d);
		glTexCoord2f(1, 0);
		glVertexV(c);
		glTexCoord2f(1, 1);
		glVertexV(b);

		glTexCoord2f(0, 1);
		glVertexV(a);
		glTexCoord2f(0, 0);
		glVertexV(d);
		glTexCoord2f(1, 1);
		glVertexV(b);
	}


	void Draw()
	{
		glEnable(GL_TEXTURE_2D);
		glPushMatrix();

		glTranslatef(Position.x, Position.y, Position.z);
		glRotatef(angle, Rotation.x, Rotation.y, Rotation.z);
		glScalef(Scale.x, Scale.y, Scale.z);



		glBindTexture(GL_TEXTURE_2D, texture.texture_id);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		glBegin(GL_TRIANGLES);
		DrawSquare(Vector(1.0f, -1.0f, 1.0f), Vector(1.0f, 1.0f, 1.0f), Vector(-1.0f, 1.0f, 1.0f), Vector(-1.0f, -1.0f, 1.0f));
		DrawSquare(Vector(-1.0f, 1.0f, -1.0f), Vector(1.0f, 1.0f, -1.0f), Vector(1.0f, -1.0f, -1.0f), Vector(-1.0f, -1.0f, -1.0f));
		DrawSquare(Vector(-1.0f, 1.0f, -1.0f), Vector(-1.0f, -1.0f, -1.0f), Vector(-1.0f, -1.0f, 1.0f), Vector(-1.0f, 1.0f, 1.0f));
		DrawSquare(Vector(1.0f, -1.0f, 1.0f), Vector(1.0f, -1.0f, -1.0f), Vector(1.0f, 1.0f, -1.0f), Vector(1.0f, 1.0f, 1.0f));
		DrawSquare(Vector(-1.0f, 1.0f, 1.0f), Vector(1.0f, 1.0f, 1.0f), Vector(1.0f, 1.0f, -1.0f), Vector(-1.0f, 1.0f, -1.0f));
		DrawSquare(Vector(1.0f, -1.0f, -1.0f), Vector(1.0f, -1.0f, 1.0f), Vector(-1.0f, -1.0f, 1.0f), Vector(-1.0f, -1.0f, -1.0f));

		glEnd();
		glPopMatrix();
		glDisable(GL_TEXTURE_2D);


	}
};

float maxf(float a, float b)
{
	if (a >= b) return a;
	return b;
}
struct Bilboard : public Object
{

	long startTime;
	Vector Direction;
	long delta;

	Bilboard()
	{

	}

	void init(long st, Vector dir)
	{
		startTime = st;
		Direction = dir;
	}


	void Draw()
	{
		glPushMatrix();
		glTranslatef(Direction.x*delta*0.01f, Direction.y*delta*0.01f, Direction.z*delta*0.01f);
		if (delta < 250) 
		{
			bilboardcolour[0] = bilboardblue[0];
			bilboardcolour[1] = bilboardblue[1];
			bilboardcolour[2] = bilboardblue[2];
			
		}
		else{
			if (delta < 500)
			{
				bilboardcolour[0] += bilboardyellow[0];
				bilboardcolour[1] += bilboardyellow[1];
				bilboardcolour[2] += bilboardyellow[2];
			}
			else
			{
				bilboardcolour[0] = bilboardred[0]*3;
				bilboardcolour[1] = bilboardred[1]*3;
				bilboardcolour[2] = bilboardred[2]*3;
				
			}
		}
		glDisable(GL_CULL_FACE);
		glEnable(GL_BLEND);
		glMaterialfv(GL_FRONT, GL_AMBIENT, bilboardcolour);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glBegin(GL_TRIANGLE_FAN);
		for (int i = 0; i < 360; i += 15)
		{
			float angle = ToRadian(i);
			glVertex3f(cos(angle), sin(angle), 0.0f);
		}
		glEnd();
		glMaterialfv(GL_FRONT, GL_AMBIENT, black);
		glDisable(GL_BLEND);
		glEnable(GL_CULL_FACE);


		glPopMatrix();
	}
};


struct Wrapper : public Object
{

	Bilboard* bilboards;
	int counter;
	bool pedal;

	Wrapper(int count = 500) : counter(count)
	{
		long currenttime = glutGet(GLUT_ELAPSED_TIME);
		pedal = false;
		bilboards = new Bilboard[counter];
		for (int i = 0; i < counter; i++)
		{
			Vector dir = Vector(((rand() / (float)RAND_MAX) - 0.5f)*0.3f, ((rand() / (float)RAND_MAX) * - 1.0f) -0.5f, ((rand() / (float)RAND_MAX) - 0.5f)*0.3f);
			bilboards[i].init(currenttime, dir);
		}
	}

	void Draw()
	{
		glScalef(0.18f, 0.08f, 0.18f);
		long currenttime = glutGet(GLUT_ELAPSED_TIME);
		for (int i = 0; i < (rand() / (float)RAND_MAX)*counter; i++)
		{
			if (pedal)
			{
				if ((currenttime - bilboards[i].startTime) < 1000)
				{
					long delta = (currenttime - bilboards[i].startTime);
					bilboards[i].delta = delta;
					bilboards[i].Draw();
				}
				else
				{
					Vector dir = Vector(((rand() / (float)RAND_MAX) - 0.5f)*0.5f, ((rand() / (float)RAND_MAX) * -1.0f) - 0.2f, ((rand() / (float)RAND_MAX) - 0.5f)*0.3f);
					bilboards[i].init(currenttime, dir);
				}
			}
			else
			{
				if ((currenttime - bilboards[i].startTime) < 1000)
				{
					long delta = (currenttime - bilboards[i].startTime);
					bilboards[i].delta = delta;
					bilboards[i].Draw();
				}
			}
		}
	}

};

struct SpaceStation : public Object
{
	CatmullRomObject base;
	Plane panel;


	SpaceStation(const Vector& pos = Vector(-0.1, 0, 0), const Vector& rot = Vector(1, 1, 1), const Vector& scale = Vector(1.3, 1.3, 1.3), const float ag = 95)
	{
		Position = pos;
		Rotation = rot;
		Scale = scale;
		angle = ag;
	}

	void Draw()
	{
		glPushMatrix();
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, panelDiff);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, panelSpec);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, panelShine);

		glTranslatef(0, 0, -1);
		glRotatef(90, 1, 0, 0);
		glTranslatef(0, 0, Position.z);
		//glTranslatef(Position.x, Position.y, Position.z);
		//glRotatef(angle, Rotation.x, Rotation.y, Rotation.z);
		glScalef(Scale.x, Scale.y, Scale.z);
		glScalef(2,2,2);

		glPushMatrix();
		glTranslatef(0.35, 0.9, 0.0);
		glRotatef(180, 1, 0, 0);
		glRotatef(90, 0, 0, 1);
		glRotatef(10, 0, 1, 1);

		glScalef(0.1, 0.4, 1);

		panel.Draw();

		glPopMatrix();

		glPushMatrix();

		glTranslatef(0, 0.45, 0);
		glRotatef(90, 1, 0, 0);
		glRotatef(90, 0, 1, 0);
		glRotatef(90, 1, 0, 0);

		glScalef(0.15, 1.5, 1);

		panel.Draw();

		glMaterialfv(GL_BACK, GL_DIFFUSE, black);
		glMaterialfv(GL_BACK, GL_SPECULAR, black);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, silverDiff);
		glMaterialfv(GL_FRONT, GL_SPECULAR, silverSpec);
		glMaterialfv(GL_FRONT, GL_SHININESS, silverShine);



		glPopMatrix();
		glDisable(GL_CULL_FACE);
		base.Draw();
		glEnable(GL_CULL_FACE);
		glPopMatrix();
	}
};

struct SpaceShip : public Object
{
	Sphere base;
	Cylinder engine;
	Wrapper wfire;
	Wrapper afire;
	Wrapper sfire;
	Wrapper dfire;

	bool w,a,s,d;
	float xmove, ymove, zmove;
	float arotate, srotate, drotate;
	float xvelocity, yvelocity, zvelocity;

	SpaceShip(const Vector& pos = Vector(0.0f, 0.0f, 0.0f), const Vector& rot = Vector(0, 0, 0), const Vector& scale = Vector(0.7, 0.7, 0.7), const float ag = 0)
	{
		Position = pos;
		Rotation = rot;
		Scale = scale;
		angle = ag;
		w = false;
		a = false;
		s = false;
		d = false;
		arotate = 0;
		srotate = 0;
		drotate = 0;
	}

	void Draw()
	{
		glPushMatrix();
		
		moveupdate();
		
	

		glTranslatef(xmove*0.05f, ymove*0.05f, zmove*0.05f);

	
		glTranslatef(Position.x, Position.y, Position.z);
		glRotatef(angle, Rotation.x, Rotation.y, Rotation.z);
		glScalef(Scale.x, Scale.y, Scale.z);



		
		glPushMatrix();
		glMaterialfv(GL_FRONT, GL_DIFFUSE, chromeDiff);
		glMaterialfv(GL_FRONT, GL_SPECULAR, chromeSpec);
		glMaterialfv(GL_FRONT, GL_SHININESS, chromeShine);
		glScalef(0.3, 0.3, 0.3);
		base.Draw();
		glPopMatrix();

		glPushMatrix();
		glMaterialfv(GL_FRONT, GL_DIFFUSE, pearlDiff);
		glMaterialfv(GL_FRONT, GL_SPECULAR, pearlSpec);
		glMaterialfv(GL_FRONT, GL_SHININESS, pearlShine);
		glTranslatef(0, 0.2, 0.0);
		glScalef(0.05, 0.4, 0.05);

		glPushMatrix();
		glRotatef(180, 1, 0, 0);
		wfire.pedal = w;
		wfire.Draw();
		glPopMatrix();

		engine.Draw();
		glPopMatrix();

		glPushMatrix();
		glMaterialfv(GL_FRONT, GL_DIFFUSE, jadeDiff);
		glMaterialfv(GL_FRONT, GL_SPECULAR, jadeSpec);
		glMaterialfv(GL_FRONT, GL_SHININESS, jadeShine);
		glTranslatef(0, -0.15, 0);
		glRotatef(50, 1, 0, 0);
		glScalef(0.05, 0.63, 0.05);

		glPushMatrix();
		sfire.pedal = s;
		sfire.Draw();
		glPopMatrix();

		engine.Draw();
		glPopMatrix();

		glPushMatrix();
		glMaterialfv(GL_FRONT, GL_DIFFUSE, turquoiseDiff);
		glMaterialfv(GL_FRONT, GL_SPECULAR, turquoiseSpec);
		glMaterialfv(GL_FRONT, GL_SHININESS, turquoiseShine);

		glTranslatef(0, -0.15, 0);
		glRotatef(120, 0, 1, 0);
		glRotatef(50, 1, 0, 0);
		glScalef(0.05, 0.63, 0.05);

		glPushMatrix();
		afire.pedal = a;
		afire.Draw();
		glPopMatrix();


		engine.Draw();
		glPopMatrix();

		glPushMatrix();
		glMaterialfv(GL_FRONT, GL_DIFFUSE, copperDiff);
		glMaterialfv(GL_FRONT, GL_SPECULAR, copperSpec);
		glMaterialfv(GL_FRONT, GL_SHININESS, copperShine);
		glTranslatef(0, -0.15, 0);
		glRotatef(240, 0, 1, 0);
		glRotatef(50, 1, 0, 0);
		glScalef(0.05, 0.63, 0.05);
		
		glPushMatrix();
		dfire.pedal = d;
		dfire.Draw();
		glPopMatrix();


		engine.Draw();
		glPopMatrix();
		
		glPopMatrix();
	}
	void moveupdate()
	{
		
		if (w)
		{
			if (yvelocity > -1.0f) yvelocity=yvelocity-0.05;
		}
		if (a)
		{
			if (yvelocity < 1.0f) yvelocity = yvelocity + 0.025;
			if (xvelocity < 1.0f) xvelocity = xvelocity + 0.0125;
			if (zvelocity > -1.0f) zvelocity = zvelocity - 0.0125;
			
			arotate++;
		}

		if (s)
		{
			if (yvelocity < 1.0f) yvelocity = yvelocity + 0.025;
			if (zvelocity < 1.0f) zvelocity = zvelocity + 0.025;

			srotate++;
		}
		if (d)
		{

			if (yvelocity < 1.0f) yvelocity = yvelocity + 0.025;
			if (xvelocity > -1.0f) xvelocity = xvelocity - 0.0125;
			if (zvelocity > -1.0f) zvelocity = zvelocity - 0.0125;

			drotate++;
		}

		xmove += xvelocity;
		ymove += yvelocity;
		zmove += zvelocity;



	}
	void Wpressed(){ w = true; }
	void Apressed(){ a = true; }
	void Spressed(){ s = true; }
	void Dpressed(){ d = true; }

	void Wreleased(){ w = false; }
	void Areleased(){ a = false; }
	void Sreleased(){ s = false; }
	void Dreleased(){ d = false; }
};



struct Camera
{
	Vector Position, viewDirection, up;
	float MoveSpeed;
	Vector MousePosition, Delta;
	float w;
	float length;
	bool direction;
	float RopeMax;
	float RopeMin;

	Camera() : Position(0.0f, 0.0f,5.0f), viewDirection(0.0f, 0.0f, -1.0f), up(0.0f, 1.0f, 0.0f)
	{
		MousePosition = Vector(0, 0, 0);
		MoveSpeed = 0.1f;
		w = 1.5f;
		length = 7.0f;
		direction = false;
		RopeMax = 7.0f;
		RopeMin = 5.0f;
	}

	void WorldToView()
	{
		Hooke();


		gluLookAt(Position.x, Position.y, Position.z,
			Position.x + viewDirection.x, Position.y + viewDirection.y, Position.z + viewDirection.z,
			up.x, up.y, up.z);
	}

	void Hooke()
	{
		if (direction)
		{
			if (Position.z <= RopeMin) direction = false;
			else
			{
				Position.z -= (glutGet(GLUT_ELAPSED_TIME) - prevTime)*0.005f;
			}
		}
		else
		{
			if (Position.z >= RopeMax) direction = true;
			else
			Position.z += (glutGet(GLUT_ELAPSED_TIME) - prevTime)*0.005f;
		}
	}

};


////////////////////// objects
SkyBox space;
Sphere planet;
Sphere aero;
Sphere sun;
SpaceStation spacestation;
SpaceShip spaceship;


void InitTextures()
{
	Color asd;
	Color orange(0.94, 0.74, 0.24);
	Color red(0.94, 0.64, 0);
	float  perlin;
	for (int y = 0; y < SizeofPlanet; y++){
		for (int x = 0; x < SizeofPlanet; x++){
			perlin = Perlin::scramble(x, y);

			PlanetP[x][y][0] = (int)perlin*0.01;
			PlanetP[x][y][1] = (int)perlin;
			PlanetP[x][y][2] = (int)255 - perlin;


			if (perlin > 0.5)
			{
				SunP[x][y][0] = (int)(orange.r * 255);
				SunP[x][y][1] = (int)(orange.g * 255);
				SunP[x][y][2] = (int)(orange.b * 255);
			}
			else
			{
				SunP[x][y][0] = (int)(red.r * 255);
				SunP[x][y][1] = (int)(red.g * 255);
				SunP[x][y][2] = (int)(red.b * 255);
			}
		}
	}

	srand(5);
	float f = rand();
	for (int i = 0; i < SizeofSB; i++){
		for (int j = 0; j < SizeofSB; j++){
			f = rand() % 100;
			if (f < 2){
				SpaceBoxP[i][j][0] = 255;
				SpaceBoxP[i][j][1] = 255;
				SpaceBoxP[i][j][2] = 200;
			}
			else {
				perlin = Perlin::scramble(i, j);
				SpaceBoxP[i][j][0] = 0 + (int)(perlin*0.009);
				SpaceBoxP[i][j][1] = 0 + (int)(perlin*0.013);
				SpaceBoxP[i][j][2] = 0 + (int)(perlin*0.015);
			}
		}
	}



}

void InitObjects()
{
	planet.textured = true;
	sun.textured = true;
	aero.transparent = true;

	sun.Position = Vector(0, 2.0f, -33.0f);
	sun.Scale = Vector(4.0f, 4.0f, 4.0f);
	aero.Position = Vector(-1.5f, -10.0f, -7.0f);
	planet.Position = Vector(-1.5f, -10.0f, -7.0f);
	planet.Scale = Vector(10.0f, 10.0f, 10.0f);
	aero.Scale = Vector(11.15f, 11.15f, 11.15f);
}

void InitScene()
{
	InitTextures();

	InitObjects();

	planet.texture.Bind(GL_RGB8, SizeofPlanet, SizeofPlanet, GL_RGB, PlanetP);
	space.texture.Bind(GL_RGB8, SizeofSB, SizeofSB, GL_RGB, SpaceBoxP);
	sun.texture.Bind(GL_RGB8, SizeofSun, SizeofSun, GL_RGB, SunP);
}


void onInitialization()
{
	glViewport(0, 0, screenWidth, screenHeight);					// Reset The Current Viewport
	glClearColor(0.0f, 0.0f, 0.5f, 0.5f);


	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix

	
	gluPerspective(60.0f, (GLfloat)screenWidth / (GLfloat)screenHeight, 0.1f, 60.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix


	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_NORMALIZE);
	glEnable(GL_CULL_FACE);


	glLightfv(GL_LIGHT0, GL_SHININESS, LightShine);
	glLightfv(GL_LIGHT0, GL_SPECULAR, LightSpec);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiff);
	glEnable(GL_LIGHT0);

	InitScene();
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


}
Camera c;
long globalelapsed;

void onDisplay() {
	globalelapsed = glutGet(GLUT_ELAPSED_TIME)*0.05f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	c.WorldToView();
	glRotatef(globalelapsed, 0, 1, 1);
	float Position[] = { 0.0f, -2.0f, -10.0f, 0.0f };

	glLightfv(GL_LIGHT0, GL_POSITION, Position);

	space.Draw();
	
	sun.Draw();
	planet.Draw();

	aero.Draw();
	spaceship.Draw();
	glRotatef(globalelapsed, 0, -1, -1);
	spacestation.Draw();


	prevTime = glutGet(GLUT_ELAPSED_TIME);
	glutSwapBuffers();
}


void onKeyboard(unsigned char key, int x, int y) {
	if (key == 'w') spaceship.Wpressed();
	if (key == 'a') spaceship.Apressed();
	if (key == 's') spaceship.Spressed();
	if (key == 'd') spaceship.Dpressed();

	glutPostRedisplay();
}


void onKeyboardUp(unsigned char key, int x, int y)
{
	if (key == 'w') spaceship.Wreleased();
	if (key == 'a') spaceship.Areleased();
	if (key == 's') spaceship.Sreleased();
	if (key == 'd') spaceship.Dreleased();
	glutPostRedisplay();
}


void onMouse(int button, int state, int x, int y) {

}


void onMouseMotion(int x, int y)
{

}


void onIdle() {
	if ((glutGet(GLUT_ELAPSED_TIME)-prevTime)>10)
	glutPostRedisplay();
}

// ...Idaig modosithatod
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A C++ program belepesi pontja, a main fuggvenyt mar nem szabad bantani
int main(int argc, char **argv) {
	glutInit(&argc, argv);                 // GLUT inicializalasa
	glutInitWindowSize(600, 600);            // Alkalmazas ablak kezdeti merete 600x600 pixel 
	glutInitWindowPosition(100, 100);            // Az elozo alkalmazas ablakhoz kepest hol tunik fel
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);    // 8 bites R,G,B,A + dupla buffer + melyseg buffer

	glutCreateWindow("Grafika hazi feladat");        // Alkalmazas ablak megszuletik es megjelenik a kepernyon

	glMatrixMode(GL_MODELVIEW);                // A MODELVIEW transzformaciot egysegmatrixra inicializaljuk
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);            // A PROJECTION transzformaciot egysegmatrixra inicializaljuk
	glLoadIdentity();

	onInitialization();                    // Az altalad irt inicializalast lefuttatjuk

	glutDisplayFunc(onDisplay);                // Esemenykezelok regisztralasa
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();                    // Esemenykezelo hurok

	return 0;
}