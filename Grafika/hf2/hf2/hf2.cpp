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
// Nev    : SRAJNER FERENC
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

#include <vector>

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...
#define MaxRayDepth 10
#define FLT_MAX 3.40282347e+038
const int screenWidth = 600;
const int screenHeight = 600;
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

	float Distance(const Vector& v){return sqrt((x - v.x)*(x - v.x) + (y - v.y)*(y - v.y) + (z - v.z)*(z - v.z)); }

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
	float ret = (float)(rand()/RAND_MAX)*max+min;
	if (ret > max) return ret-max+min;
	return ret;
}

struct Ray
{
	Vector origin, direction;
	float mint, maxt;
	Ray()
	{
		origin = direction = Vector(0, 0, 0);
		mint = 0.01;
		maxt = FLT_MAX;
	}
	Ray(Vector o, Vector d) : origin(o), direction(d.normalize())    {
		mint = 0.01;
		maxt = FLT_MAX;
	}

	Ray(Vector o, Vector d, float max) : origin(o), direction(d.normalize()), maxt(max) { mint = 0.01; }

	Vector operator*(const float t) const
	{
		return origin + (direction * t);
	}
};

struct Material
{
	Material(){}

	virtual Color GetColor(Vector hitPoint, Vector hitNormal, Ray V, int depth) = 0;
};

struct Object
{

	Material* material;

	Object(){}

	Object(Material* m) : material(m)
	{

	}

	virtual float intersect(Ray ray, Vector* Normal = NULL, Vector* hPoint = NULL) = 0;

	virtual void ICanHazCactus(Vector* Normal = NULL, Vector* hPoint = NULL) = 0;

	bool Quadratic(float A, float B, float C, float *t0, float *t1) {
		float discrim = B * B - 4.0f * A * C;
		if (discrim <= 0.0) return false;
		float rootDiscrim = sqrtf(discrim);


		*t0 = (-B + rootDiscrim) / 2 / A;
		*t1 = (-B - rootDiscrim) / 2 / A;

		if (*t0 > *t1)
		{
			float t = *t0;
			*t0 = *t1;
			*t1 = t;
		}
		return true;
	}
};

struct PointLight
{
	Vector Center;
	Color EmissionColor;
	float intensity;

	PointLight()
	{
		Center = Vector(0, 0, 0);
		EmissionColor = Color(0, 0, 0);
		intensity = 121.0f;
	}
	PointLight(Vector c, Color ec)
	{
		Center = c; EmissionColor = ec;
		intensity = 1210.0f;
	}

	Color GetLightColor(Vector& point)
	{
		float distance = Center.Distance(point);

		return EmissionColor*	(((intensity - distance)*(intensity - distance)) / (intensity*intensity));
	}

};

int NumberofObjects = 12;
int NumberofLights = 3;
Object** Objects;
PointLight* Lights;

Color bgColor = Color(0.7, 0.94, 1.0);

Color image[screenWidth*screenHeight];

struct Matrix
{
public:
	//source: ftp://download.intel.com/design/PentiumIII/sml/24504301.pdf
	//		  http://www.unknownroad.com/rtfm/graphics/rt_normals.html
	//		  http://www.inversereality.org/tutorials/graphics%20programming/3dwmatrices.html

	float* matrix;
	float* inverse;

	float determinant;

	bool isdet;
	bool isinv;


	Matrix()
	{
		matrix = new float[16];
		isdet = false;
		isinv = false;
	}

	Matrix(float values[16])
	{
		matrix = new float[16];
		for (int i = 0; i < 16; i++)
			matrix[i] = values[i];
	}

	void Init(Vector pos, Vector rot, Vector scale)
	{
		// source: http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/

		float x = pos.x;
		float y = pos.y;
		float z = pos.z;
		float sinx = sinf(ToRadian(rot.x));
		float siny = sinf(ToRadian(rot.y));
		float sinz = sinf(ToRadian(rot.z));
		float cosx = cosf(ToRadian(rot.x));
		float cosy = cosf(ToRadian(rot.y));
		float cosz = cosf(ToRadian(rot.z));
		float scalex = scale.x;
		float scaley = scale.y;
		float scalez = scale.z;


		matrix[0] = scalex*(cosy*cosz);
		matrix[1] = scaley*(sinx*siny*cosz - cosx*sinz);
		matrix[2] = scalez*(cosx*siny*cosz + sinx*sinz);
		matrix[3] = x;
		matrix[4] = scalex*(cosy*sinz);
		matrix[5] = scaley*(cosx*cosz + sinx*siny*sinz);
		matrix[6] = scalez*-(sinx*cosz + cosx*siny*sinz);
		matrix[7] = y;
		matrix[8] = scalex*(-siny);
		matrix[9] = scaley*(sinx*cosy);
		matrix[10] = scalez*cosx*cosy;
		matrix[11] = z;
		matrix[12] = 0;
		matrix[13] = 0;
		matrix[14] = 0;
		matrix[15] = 1;



	}

	float*     GetInverseMatrix()
	{
		if (!isinv)
		{
			inverse = new float[16];
			float tmp[12];
			float src[16]; 
			float det; 
		
			for (int i = 0; i < 4; i++) {
				src[i] = matrix[i * 4];
				src[i + 4] = matrix[i * 4 + 1];
				src[i + 8] = matrix[i * 4 + 2];
				src[i + 12] = matrix[i * 4 + 3];
			}
			tmp[0] = src[10] * src[15];
			tmp[1] = src[11] * src[14];
			tmp[2] = src[9] * src[15];
			tmp[3] = src[11] * src[13];
			tmp[4] = src[9] * src[14];
			tmp[5] = src[10] * src[13];
			tmp[6] = src[8] * src[15];
			tmp[7] = src[11] * src[12];
			tmp[8] = src[8] * src[14];
			tmp[9] = src[10] * src[12];
			tmp[10] = src[8] * src[13];
			tmp[11] = src[9] * src[12];

			inverse[0] = tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7];
			inverse[0] -= tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7];
			inverse[1] = tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7];
			inverse[1] -= tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7];
			inverse[2] = tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7];
			inverse[2] -= tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7];
			inverse[3] = tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6];
			inverse[3] -= tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6];
			inverse[4] = tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3];
			inverse[4] -= tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3];
			inverse[5] = tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3];
			inverse[5] -= tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3];
			inverse[6] = tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3];
			inverse[6] -= tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3];
			inverse[7] = tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2];
			inverse[7] -= tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2];

			tmp[0] = src[2] * src[7];
			tmp[1] = src[3] * src[6];
			tmp[2] = src[1] * src[7];
			tmp[3] = src[3] * src[5];
			tmp[4] = src[1] * src[6];
			tmp[5] = src[2] * src[5];

			tmp[6] = src[0] * src[7];
			tmp[7] = src[3] * src[4];
			tmp[8] = src[0] * src[6];
			tmp[9] = src[2] * src[4];
			tmp[10] = src[0] * src[5];
			tmp[11] = src[1] * src[4];

			inverse[8] = tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15];
			inverse[8] -= tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15];
			inverse[9] = tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15];
			inverse[9] -= tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15];
			inverse[10] = tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15];
			inverse[10] -= tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15];
			inverse[11] = tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14];
			inverse[11] -= tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14];
			inverse[12] = tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9];
			inverse[12] -= tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10];
			inverse[13] = tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10];
			inverse[13] -= tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8];
			inverse[14] = tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8];
			inverse[14] -= tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9];
			inverse[15] = tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9];
			inverse[15] -= tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8];

			det = src[0] * inverse[0] + src[1] * inverse[1] + src[2] * inverse[2] + src[3] * inverse[3];

			determinant = det;

			det = 1 / det;
			for (int j = 0; j < 16; j++)
				inverse[j] *= det;
			isinv = true;
			isdet = true;
		}
		return inverse;


	}



	void SetMatrix(float* mat)
	{
		for (int i = 0; i < 16; i++) matrix[i] = mat[i];
	}

	void	GetTransposeMatrix(float* transpose)
	{	
			for (int i = 0; i < 4; i++)
			{
				transpose[4 * i]	 = matrix[i];
				transpose[4 * i + 1] = matrix[i + 4];
				transpose[4 * i + 2] = matrix[i + 8];
				transpose[4 * i + 3] = matrix[i + 12];

			}
	}

	~Matrix()
	{
		delete[] matrix;
		if (isinv)delete[] inverse;
	}

};

struct Vector4
{
	float vector4[4];


	Vector4(Vector v, float vf)
	{
		vector4[0] = v.x; vector4[1] = v.y; vector4[2] = v.z; vector4[3] = vf;
	}
	Vector4()
	{

	}
	Vector operator*(const Matrix& V)
	{
		float x = 0, y = 0, z = 0;

		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 4; i++){
				if (j == 0) x += V.matrix[4 * j + i] * vector4[i];
				if (j == 1) y += V.matrix[4 * j + i] * vector4[i];
				if (j == 2) z += V.matrix[4 * j + i] * vector4[i];
			}
		}
		return Vector(x, y, z);
	}

	Vector operator%(Matrix& V)
	{
		float* temp = new float[16];
		V.GetTransposeMatrix(temp);
			float x = 0, y = 0, z = 0;

			for (int j = 0; j < 3; j++){
				for (int i = 0; i < 4; i++)
			{
					if (j == 0) x += temp[4 * j + i] * vector4[i];
					if (j == 1) y += temp[4 * j + i] * vector4[i];
					if (j == 2) z += temp[4 * j + i] * vector4[i];
			}
	}
			delete temp;
		return Vector(x, y, z);
	}

	~Vector4()
	{
	}

};

struct Scene
{
	Scene()
	{
	}

	static Object* IntersectAll(Ray ray, Vector* Point, Vector* Normal)
	{
		float tnear = FLT_MAX;
		float dist;
		Object* target = NULL;

		for (int i = 0; i < NumberofObjects; ++i) {

			dist = Objects[i]->intersect(ray, Normal, Point);
			if (dist < tnear && dist<ray.maxt && dist>ray.mint)
			{
				target = &*Objects[i];
				tnear = dist;
			}
		}
		return target;
	}

	static Color Trace(Ray ray, int depth)
	{
		if (depth > MaxRayDepth) return bgColor;

		Vector* hitPoint = new Vector(0, 0, 0);
		Vector* hitNormal = new Vector(0, 0, 0);
		Object* target = IntersectAll(ray, hitPoint, hitNormal);

		if (target == NULL)
		{
			delete hitNormal;
			delete hitPoint;
			return bgColor;
		}
		Color r = (target->material->GetColor(*hitPoint, *hitNormal, ray, depth + 1));
		delete hitNormal;
		delete hitPoint;

		return r;
	}
};

struct Camera
{
	// https://wiki.sch.bme.hu/Számítógépes_grafika_házi_feladat_tutorial#A_harmadik_h.C3.A1zihoz_sz.C3.BCks.C3.A9ges_elm.C3.A9let

	Vector cameraPos;
	Vector cameraDir;
	Vector up;
	Vector right;

	float FieldofView;

	Camera(){}
	Camera(Vector pos, Vector target, Vector u, float fov)
	{
		cameraPos = pos;
		cameraDir = target - pos;
		cameraDir.normalize();

		FieldofView = tanf((fov*M_PI / 180) / 2);


		right = ((cameraDir) % u).normalize();
		up = (right % (cameraDir)).normalize();

	}

	void Draw()
	{
		Color pixel;

		for (float x = 0; x < (float)screenWidth; x++)
		{
			for (float y = 0; y < (float)screenHeight; y++)
			{
				float x2 = (x - screenWidth / 2.0f) / (screenWidth / 2.0f / FieldofView);
				float y2 = (y - screenWidth / 2.0f) / (screenHeight / 2.0f / FieldofView);

				Vector v = cameraDir + right*x2 + up*y2;

				Ray r = Ray(cameraPos, (v).normalize());

				if (r.direction.z < 0)
				{

				}
				pixel = Scene::Trace(r, 0);
				image[(int)y*screenWidth + (int)x] = pixel;
			}

		}
	}
};

Camera* camera;

struct BlinnPhong : Material
{
	// http://www.codeproject.com/Articles/785084/A-generic-lattice-noise-algorithm-an-evolution-of
	// http://www.sorgonet.com/linux/noise_textures/

	Color diffuse;
	Color specular;
	float Phonglevel;

	BlinnPhong(Color kd, Color ks)
	{
		diffuse = kd;
		specular = ks;
		Phonglevel = 1000;
	}

	float InterPolation(float a, float b, float c)
	{
		return a + (b - a)*c*c*(3 - 2 * c);
	}

	float InterLinear(float a, float b, float c)
	{
		return a*(1 - c) + b*c;
	}

	float Noise(int x)
	{
		x = (x << 13) ^ x;
		return (((x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0);
	}

	float PerlinNoise(float x, float y, int width, int octaves, int seed, double periode){
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

	float scramble(Vector hP)
	{
		int x = abs(hP.x*1000);
		int y = abs(hP.y*1000);
		int seed;
		int width;
		float  disp1, disp2, disp3, disp4, disp5, disp6, scale;


		scale = 10;
		width = 12325;
		seed  = 10;

		disp1 = PerlinNoise(x*scale, y*scale, width, 1, seed, 100);
		disp2 = PerlinNoise(x*scale, y*scale, width, 2, seed, 50);
		disp3 = PerlinNoise(x*scale, y*scale, width, 3, seed, 12.5);
		disp4 = PerlinNoise(x*scale, y*scale, width, 4, seed, 6.25);
		disp5 = PerlinNoise(x*scale, y*scale, width, 5, seed, 3.125);
		disp6 = PerlinNoise(x*scale, y*scale, width, 6, seed, 1.56);

		float valor = (disp1)+(disp2*0.5) + (disp3*0.5) + (disp4*0.125) + (disp5*0.03125) + (disp6*0.0156);
	
		return float(InterLinear(valor, 0, 0)/150.0f);
	}

	Color GetColor(Vector hitPoint, Vector hitNormal, Ray V, int depth)
	{
		if (depth > MaxRayDepth) return bgColor;
		Color ret = Color(0, 0, 0);
		float costheta;
		float cosdelta;
		Color Light;

		for (int i = 0; i < NumberofLights; ++i)
		{
			Vector L = Lights[i].Center - hitPoint;
			L.normalize();

			costheta = hitNormal * L;
			if (costheta < 0) break;

			Object* target = (Scene::IntersectAll(Ray(hitPoint, L, hitPoint.Distance(Lights[i].Center)), NULL, NULL));
			if (target != NULL) break;

			Light= Lights[i].GetLightColor(hitPoint);
			ret += Light *diffuse * costheta *scramble(hitPoint);

			Vector H = L + V.direction - hitPoint;
			H.normalize();
			cosdelta = hitNormal * H;
			if (cosdelta < 0) break;

			ret +=Light * specular * max(0, pow(cosdelta, Phonglevel));
		}
		return ret;
	}
};

struct Fresnel : Material
{
	Color refraction; 
	Color extinction; 
	bool isreflective;
	bool isrefractive;

	Color F0;

	Fresnel(Color n, Color k, bool reflect, bool refract) : refraction(n), extinction(k), isreflective(reflect), isrefractive(refract)
	{
		F0 = ((refraction - 1)*(refraction - 1) + extinction*extinction) / ((refraction + 1)*(refraction + 1) + extinction*extinction);
	}

	Vector  ReflectionDir(Vector& N, Vector& V) {
		float cosa = -(N* V);
		return (V + N * cosa * 2).normalize();
	}

	Vector RefractionDir(Vector& N, Vector& V) {

		float cosa = -(N* V), cn = refraction.r;
		if (cosa < 0)
		{
			cosa = -cosa;
			N = (N*-1);
			cn = 1 / refraction.r;
		}
		float disc = 1 - (1 - cosa * cosa) / cn / cn;
		if (disc < 0) return Vector(0,0,0);
		return (V / cn + N * (cosa / cn - sqrt(disc))).normalize();
	}

	Color GetFresnel(Vector& N, Vector& V)
	{
		float cosa = fabs(N * V);
		return  F0 + (Color(1, 1, 1) - F0) * pow(1 - cosa, 5);
	}

	Color GetColor(Vector hitPoint, Vector hitNormal, Ray V, int depth)
	{
		if (depth > MaxRayDepth) return bgColor;

		Color color = Color(0, 0, 0);

		Color Fresnel = GetFresnel(hitNormal, V.direction);
		
		if (isreflective)
		{
			color += Fresnel * Scene::Trace(Ray(hitPoint, ReflectionDir(hitNormal, V.direction)), depth + 1);
		}

		if (isrefractive)
		{
			color += (Color(1, 1, 1) - Fresnel) *Scene::Trace(Ray(hitPoint, RefractionDir(hitNormal, V.direction)), depth + 1);
		}

		return color;
	}
};
struct Plane : Object
{
	Vector normal;
	Vector pos;

	Plane(Vector p, Vector n, Material* m) : Object(m)
	{
		normal = n; pos = p;
		normal.normalize();
	}

	float intersect(Ray ray, Vector* Normal = NULL, Vector* Point = NULL)
	{
		float t;


		t = (ray.origin - pos)*normal;
		t /= ray.direction*normal;
		t = -t;

		if (t > ray.maxt || t < ray.mint) return false;
		if (Normal != NULL) *Normal = normal;
		if (Point != NULL) *Point = Vector(ray*t);
		return t;
	}

	void ICanHazCactus(Vector* Normal = NULL, Vector* hPoint = NULL)
	{

	}
};

struct Paraboloid : public Object
{
	Matrix WorldToObject;
	Matrix ObjectToWorld;

	Paraboloid(Vector Position, Vector Rotation, Vector Scale, Material* m) : Object(m)
	{
		ObjectToWorld.Init(Position, Rotation, Scale);
		WorldToObject.SetMatrix(ObjectToWorld.GetInverseMatrix());

	}

	float intersect(Ray r, Vector* Normal = NULL, Vector* hPoint = NULL)
	{
		Ray ray;
		ray.direction = Vector4(r.direction, 0) *WorldToObject;
		ray.origin = Vector4(r.origin, 1)*WorldToObject;
		
		Vector hitPoint;
		Vector hitNormal;

		float distancefromParaboloid = FLT_MAX;
		float distancefromCircle = FLT_MAX;
		bool hitCircle = false;
		bool hitParaboloid = false;

		//	http://www.csie.ntu.edu.tw/~cyy/courses/rendering/pbrt-2.00/html/classParaboloid.html

		float A = (ray.direction.x*ray.direction.x + ray.direction.y* ray.direction.y);
		float B = 2 * (ray.direction.x * ray.origin.x + ray.direction.y * ray.origin.y) - ray.direction.z;
		float C = (ray.origin.x * ray.origin.x + ray.origin.y * ray.origin.y) - ray.origin.z;
		float t0, t1;

		Quadratic(A, B, C, &t0, &t1);

		if (t0 < ray.mint && t1>ray.mint) distancefromParaboloid = t1;
		else if (t0 > ray.mint) distancefromParaboloid = t0;

		hitPoint = ray*distancefromParaboloid;

		if (hitPoint.z > 0 && hitPoint.z < 1)
		{
			hitParaboloid = true;
			hitNormal = Vector(hitPoint.x, hitPoint.y, -0.5);
			hitNormal.normalize();
		}
		else
		{
			float dist;

			dist = (ray.origin - Vector(0, 0, 1))*Vector(0, 0, 1);
			dist /= ray.direction*Vector(0, 0, 1);
			dist = -dist;
			if (dist > 0)
			{
				distancefromCircle = dist;
				hitPoint = Vector(ray*distancefromCircle);
				if (hitPoint.x*hitPoint.x + hitPoint.y*hitPoint.y < 1)
				{
					hitCircle = true;
					hitNormal = Vector(0, 0, 1);
				}
			}
		}
		if (!hitCircle) hitPoint = Vector(ray*distancefromParaboloid);


		if (hitPoint.Distance(ray.origin) < ray.mint || hitPoint.Distance(ray.origin) > ray.maxt)
		{
			return -FLT_MAX;
		}
		if (!hitParaboloid &&!hitCircle) return -FLT_MAX;

		if (Normal != NULL)*Normal = Vector4(hitNormal, 0)%WorldToObject;
		if (hPoint != NULL)*hPoint = Vector4(hitPoint, 1) *ObjectToWorld;

		if (hitCircle) return distancefromCircle;
		else return distancefromParaboloid;
	}

	void ICanHazCactus(Vector* Normal = NULL, Vector* hPoint = NULL)
	{
		Vector randomPoint = Vector(random(2, 10), random(1, 10), random(0.5, 0.8));
		Vector Direction = (Vector(0, 0, random(0.5, 0.8)) - randomPoint).normalize();
		Ray ray = Ray(randomPoint, Direction);

		float distance;

		float A = (ray.direction.x*ray.direction.x + ray.direction.y* ray.direction.y);
		float B = 2 * (ray.direction.x * ray.origin.x + ray.direction.y * ray.origin.y) - ray.direction.z;
		float C = (ray.origin.x * ray.origin.x + ray.origin.y * ray.origin.y) - ray.origin.z;

		float t1;
		Quadratic(A, B, C, &distance, &t1);

		Vector hitPoint = ray*distance;

		if (Normal != NULL)*Normal = Vector4(Vector(hitPoint.x,hitPoint.y, -0.5).normalize(), 0)%WorldToObject;
		if (hPoint != NULL)*hPoint = Vector4(hitPoint, 1)*ObjectToWorld;


	}

};

struct Cylinder : public Object
{
	Matrix WorldToObject;
	Matrix ObjectToWorld;
	
	Cylinder(Vector Position, Vector Rotation, Vector Scale, Material* m) : Object(m)
	{
		ObjectToWorld.Init(Position, Rotation, Scale);
		WorldToObject.SetMatrix(ObjectToWorld.GetInverseMatrix());		
	}

	float intersect(Ray r, Vector* Normal = NULL, Vector* hPoint = NULL)
	{
		Vector hitPoint;
		Vector hitNormal;

		Ray ray;
		ray.direction = Vector4(r.direction, 0)*WorldToObject;
		ray.origin = Vector4(r.origin, 1)*WorldToObject;


		float distancefromCylinder = FLT_MAX;
		float distancefromCircle = FLT_MAX;

		bool hitCircles = false;

		// http://www.csie.ntu.edu.tw/~cyy/courses/rendering/pbrt-2.00/html/classCylinder.html

		float A = ray.direction.x*ray.direction.x + ray.direction.y*ray.direction.y;
		float B = 2 * (ray.direction.x*ray.origin.x + ray.direction.y*ray.origin.y);
		float C = ray.origin.x*ray.origin.x + ray.origin.y*ray.origin.y - 1;

		float t0, t1;
		if (Quadratic(A, B, C, &t0, &t1))
		{

			if (t0 < ray.mint && t1>ray.mint) distancefromCylinder = t1;
			else if (t0>ray.mint) distancefromCylinder = t0;

			hitPoint = ray*distancefromCylinder;
		}

		if (hitPoint.z > 0 && hitPoint.z  < 1)
		{
			hitNormal = Vector(hitPoint.x, hitPoint.y, 0);
			hitNormal.normalize();
		}

		else
		{
			float dist;

			dist = (ray.origin - Vector(0, 0, 1))*Vector(0, 0, 1);
			dist /= ray.direction*Vector(0, 0, 1);
			dist = -dist;
			if (dist > 0)
			{
				distancefromCircle = dist;
				hitPoint = Vector(ray*distancefromCircle);
				if (hitPoint.x*hitPoint.x + hitPoint.y*hitPoint.y < 1)
				{
					hitCircles = true;
					hitNormal = Vector(0, 0, 1);
				}
			}

			dist = (ray.origin - Vector(0, 0, 0))*Vector(0, 0, -1);
			dist /= ray.direction*Vector(0, 0, -1);
			dist = -dist;
			if (dist > 0 && dist < distancefromCircle)
			{
				hitPoint = Vector(ray*dist);

				if (hitPoint.x*hitPoint.x + hitPoint.y*hitPoint.y < 1)
				{
					hitCircles = true;
					distancefromCircle = dist;
					hitPoint = Vector(ray*distancefromCircle);
					hitNormal = Vector(0, 0, -1);
				}
			}
			if (!hitCircles) return -FLT_MAX;
		}

		if (hitPoint.Distance(ray.origin) < ray.mint || hitPoint.Distance(ray.origin) > ray.maxt)
		{
			return -FLT_MAX;
		}

		if (Normal != NULL)*Normal = Vector4(hitNormal, 0) % WorldToObject;// *ObjectToWorld;
		if (hPoint != NULL)*hPoint = Vector4(hitPoint, 1)*ObjectToWorld;

		if (hitCircles) return distancefromCircle;
		else return distancefromCylinder;
	}

	void ICanHazCactus(Vector* Normal = NULL, Vector* hPoint = NULL)
	{
		Vector randomPoint = Vector(random(2, 10), random(1, 10), random(0.5, 0.8));
		Vector Direction = (Vector(0, 0, random(0.5, 0.8)) - randomPoint).normalize();
		
		Ray ray = Ray(randomPoint, Direction);

		float distance;

		float A = ray.direction.x*ray.direction.x + ray.direction.y*ray.direction.y;
		float B = 2 * (ray.direction.x*ray.origin.x + ray.direction.y*ray.origin.y);
		float C = ray.origin.x*ray.origin.x + ray.origin.y*ray.origin.y - 1;

		float t1;
		Quadratic(A, B, C, &distance, &t1);

		Vector hitPoint = ray*distance;

		if (Normal != NULL)*Normal = Vector4(Vector(hitPoint.x, hitPoint.y, 0).normalize(), 0)%WorldToObject;
		if (hPoint != NULL)*hPoint = Vector4(hitPoint, 1)*ObjectToWorld;


	}

};

struct Ellipsoid : public Object
{
	Matrix WorldToObject;
	Matrix ObjectToWorld;

	Ellipsoid(Vector Position, Vector Rotation, Vector Scale, Material* m) : Object(m)    {

		ObjectToWorld.Init(Position, Rotation, Scale);
		WorldToObject.SetMatrix(ObjectToWorld.GetInverseMatrix());
	}


	float intersect(Ray r, Vector* Normal = NULL, Vector* hPoint = NULL)
	{
		Ray ray;
		ray.direction = Vector4(r.direction, 0) *WorldToObject;
		ray.origin = Vector4(r.origin, 1)*WorldToObject;

		float distancefromSphere = FLT_MAX;

		Vector hitPoint;


		// http://www.csie.ntu.edu.tw/~cyy/courses/rendering/pbrt-2.00/html/classSphere.html

		float A = ray.direction.x*ray.direction.x + ray.direction.y*ray.direction.y + ray.direction.z*ray.direction.z;
		float B = 2 * (ray.direction.x*ray.origin.x + ray.direction.y*ray.origin.y + ray.direction.z*ray.origin.z);
		float C = ray.origin.x*ray.origin.x + ray.origin.y*ray.origin.y + ray.origin.z*ray.origin.z - 1;

		float t0, t1;
		if (!Quadratic(A, B, C, &t0, &t1))
			return -FLT_MAX;


		if (t0 < ray.mint && t1>ray.mint) distancefromSphere = t1;
		else if (t0>ray.mint) distancefromSphere = t0;

	
		hitPoint = ray*distancefromSphere;

		if (hitPoint.Distance(ray.origin) < ray.mint || hitPoint.Distance(ray.origin) > ray.maxt)
		{
			return -FLT_MAX;
		}


		if (Normal != NULL)*Normal = Vector4(hitPoint, 0)%WorldToObject;
		if (hPoint != NULL)*hPoint = Vector4(hitPoint, 1)*ObjectToWorld;

		return distancefromSphere;
	}

	void ICanHazCactus(Vector* Normal = NULL, Vector* hPoint = NULL)
	{
		Vector randomPoint = Vector(random(2, 10), random(1, 10), random(0.5, 0.8));
		Vector a = Vector(0, 0, random(0.5, 0.8));
		Vector Direction = (a - randomPoint).normalize();
		Ray ray = Ray(randomPoint, Direction);

		float distance;

		float A = ray.direction.x*ray.direction.x + ray.direction.y*ray.direction.y + ray.direction.z*ray.direction.z;
		float B = 2 * (ray.direction.x*ray.origin.x + ray.direction.y*ray.origin.y + ray.direction.z*ray.origin.z);
		float C = ray.origin.x*ray.origin.x + ray.origin.y*ray.origin.y + ray.origin.z*ray.origin.z - 1;

		float t1;
		Quadratic(A, B, C, &distance, &t1);
	
		Vector hitPoint = ray*distance;

		if (Normal != NULL) *Normal = Vector4(hitPoint, 0)%WorldToObject;
		if (hPoint != NULL) *hPoint = Vector4(hitPoint, 1)*ObjectToWorld;
	}
};

struct EllipsoidCacti
{
	Material* gold;

	EllipsoidCacti()
	{
		gold = new Fresnel(Color(0.17, 0.35, 1.5), Color(3.1, 2.7, 1.9), true, false);
	}
	void GenerateStub(int numberofobject, int number, Vector Pos, Vector Scale, Vector Trans)
	{
		if (number > 4 || numberofobject==NumberofObjects) return;
		Objects[numberofobject] = new Ellipsoid(Pos, Trans, Scale,gold);
		Vector* normal= new Vector(0,0,0);
		Vector* pos = new Vector(0,0,0);

		Objects[numberofobject]->ICanHazCactus(normal, pos);

			float x = atan2(normal->x,normal->y);
			float y = atan2(normal->y,normal->z);
			float z = atan2(normal->z,normal->x);
			GenerateStub(numberofobject + 1, number + 1, *pos, Scale*0.5, Vector(ToDegree(x), ToDegree(y), ToDegree(z)));
	}
};

struct ParaboloidCacti
{
	Material* silver;

	ParaboloidCacti()
	{
		silver = new Fresnel(Color(0.14, 0.16, 0.13), Color(4.1, 2.3, 3.1), true, false);
	}
	void GenerateStub(int numberofobject, int number, Vector Pos, Vector Scale, Vector Trans)
	{
		if (number > 4 || numberofobject == NumberofObjects) return;
		Objects[numberofobject] = new Paraboloid(Pos, Trans, Scale, silver);
		Vector* normal = new Vector(0, 0, 0);
		Vector* pos = new Vector(0, 0, 0);

		Objects[numberofobject]->ICanHazCactus(normal, pos);

		float x = atan2(normal->x, normal->y);
		float y = atan2(normal->y, normal->z);
		float z = atan2(normal->z, normal->x);
		GenerateStub(numberofobject + 1, number + 1, *pos, Scale*0.5, Vector(ToDegree(x), ToDegree(y), ToDegree(z)));
	}
};

struct CylinderCacti
{
	Material* glass;

	CylinderCacti()
	{
		glass = new Fresnel(Color(1.5, 1.5, 1.5), Color(0, 0, 0), true, true);
	}
	void GenerateStub(int numberofobject, int number, Vector Pos, Vector Scale, Vector Trans)
	{
		if (number > 4 || numberofobject == NumberofObjects) return;
		Objects[numberofobject] = new Cylinder(Pos, Trans, Scale, glass);
		Vector* normal = new Vector(0, 0, 0);
		Vector* pos = new Vector(0, 0, 0);

		Objects[numberofobject]->ICanHazCactus(normal, pos);


		float x = atan2(normal->x, normal->y);
		float y = atan2(normal->y, normal->z);
		float z = atan2(normal->z, normal->x);
		GenerateStub(numberofobject + 1, number + 1, *pos, Scale*0.5, Vector(ToDegree(x), ToDegree(y), ToDegree(z)));
	}
};


// Inicializacio, a program futasanak kezdeten, az OpenGL kontextus letrehozasa utan hivodik meg (ld. main() fv.)
void onInitialization() {
	glViewport(0, 0, screenWidth, screenHeight);

	camera = new Camera(Vector(-3,-5,3.5), Vector(0, 0, 2), Vector(0, 0, 1), 90.0f);

	Objects = new  Object*[NumberofObjects];

	Material* desk = new BlinnPhong(Color(0.5, 0.3, 0.0), Color(0.5, 0.5, 0.5));


	Objects[0] = new  Plane(Vector(0,0,-0.5), Vector(0, 0, 1), desk);


	EllipsoidCacti ec;
	ec.GenerateStub(1, 0, Vector(-3 ,0 ,2),Vector(1,1,2), Vector(0,0,0));
	ParaboloidCacti pc;
	pc.GenerateStub(4, 0, Vector(0.5, -2, 0), Vector(1, 1, 2), Vector(0, 0, 0));
	CylinderCacti cc;
	cc.GenerateStub(7, 0, Vector(1, 0, 0), Vector(1, 1, 2), Vector(0, 0, 0));

	Lights = new PointLight[NumberofLights];

	Lights[0] = PointLight(Vector(0, 0, 3), Color(1.0, 0.0, 0.0));

	Lights[1] = PointLight(Vector(1,3,1), Color(0.0, 1.0, 0.0));

	Lights[2] = PointLight(Vector(0, -5, 0), Color(0.0, 0.0, 1.0));
}

// Rajzolas, ha az alkalmazas ablak ervenytelenne valik, akkor ez a fuggveny hivodik meg
void onDisplay() {
	glClearColor(0.1f, 0.2f, 0.3f, 1.0f);        // torlesi szin beallitasa
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // kepernyo torles

	camera->Draw();
	glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, image);


	glutSwapBuffers();                     // Buffercsere: rajzolas vege

}

// Billentyuzet esemenyeket lekezelo fuggveny (lenyomas)
void onKeyboard(unsigned char key, int x, int y) {
	if (key == 'd') glutPostRedisplay();         // d beture rajzold ujra a kepet

}

// Billentyuzet esemenyeket lekezelo fuggveny (felengedes)
void onKeyboardUp(unsigned char key, int x, int y) {

}

// Eger esemenyeket lekezelo fuggveny
void onMouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)   // A GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON illetve GLUT_DOWN / GLUT_UP
		glutPostRedisplay();                          // Ilyenkor rajzold ujra a kepet
}
// Eger mozgast lekezelo fuggveny
void onMouseMotion(int x, int y)
{

}

// `Idle' esemenykezelo, jelzi, hogy az ido telik, az Idle esemenyek frekvenciajara csak a 0 a garantalt minimalis ertek
void onIdle() {

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