/* Lab 4 base code - transforms using local matrix functions 
   to be written by students - <2019 merge with tiny loader changes to support multiple objects>
	CPE 471 Cal Poly Z. Wood + S. Sueda
*/

#include <iostream>
#include <glad/glad.h>

#include "GLSL.h"
#include "Program.h"
#include "Shape.h"
#include "MatrixStack.h"
#include "WindowManager.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

// value_ptr for glm
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace std;
using namespace glm;

class Application : public EventCallbacks
{

public:

	WindowManager * windowManager = nullptr;

	// Our shader program
	std::shared_ptr<Program> prog;

    // Our shader program
	std::shared_ptr<Program> solidColorProg;

	// Shape to be used (from  file) - modify to support multiple
    vector<shared_ptr<Shape>> carMesh;
    vector<shared_ptr<Shape>> houseMesh;
    vector<shared_ptr<Shape>> santaMesh;
    vector<shared_ptr<Shape>> sleighMesh;
    shared_ptr<Shape> floorMesh;
    shared_ptr<Shape> lampMesh;
    shared_ptr<Shape> treeMesh;

	// Contains vertex information for OpenGL
	GLuint VertexArrayID;

	// Data necessary to give our triangle to OpenGL
	GLuint VertexBufferID;

	//example data that might be useful when trying to compute bounds on multi-shape
	vec3 santaMin;
    vec3 santaMax;
    vec3 sleighMin;
    vec3 sleighMax;
    vec3 houseMin;
    vec3 houseMax;
    vec3 carMin;
    vec3 carMax;

    // animation variable
    float driveTheta = 0;
    float sceneRotate = -0.3;

	void printMat(float *A, const char *name = 0)
	{
   // OpenGL uses col-major ordering:
   // [ 0  4  8 12]
   // [ 1  5  9 13]
   // [ 2  6 10 14]
   // [ 3  7 11 15]
   // The (i,j)th element is A[i+4*j].
		if(name) {
			printf("%s=[\n", name);
		}
		for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
				printf("%- 5.2f ", A[i+4*j]);
			}
			printf("\n");
		}
		if(name) {
			printf("];");
		}
		printf("\n");
	}

	void createIdentityMat(float *M)
	{
	//set all values to zero
		for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
				M[i*4+j] = 0;
			}
		}
	//overwrite diagonal with 1s
		M[0] = M[5] = M[10] = M[15] = 1;
	}

	void createTranslateMat(float *T, float x, float y, float z)
	{
        //set all values to zero
        for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
				T[i*4+j] = 0;
			}
		}

        //overwrite diagonal with 1s
        T[0] = T[5] = T[10] = T[15] = 1;

        T[12] = x;
        T[13] = y;
        T[14] = z;
	}


	void createScaleMat(float *S, float x, float y, float z)
	{
        //set all values to zero
        for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
				S[i*4+j] = 0;
			}
		}

        //overwrite diagonal with values and 1
        S[0] = x;
        S[5] = y;
        S[10] = z;
        S[15] = 1;
	}

	void createRotateMatX(float *R, float radians)
	{ 
        //set all values to zero
        for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
				R[i*4+j] = 0;
			}
		}

        R[0] = 1;
        R[5] = cos(radians);
        R[6] = sin(radians);
        R[9] = -sin(radians);
        R[10] = cos(radians);
        R[15] = 1;
	}

	void createRotateMatY(float *R, float radians)
	{
        //set all values to zero
        for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
				R[i*4+j] = 0;
			}
		}

        R[0] = cos(radians);
        R[2] = -sin(radians);
        R[5] = 1;
        R[8] = sin(radians);
        R[10] = cos(radians);
        R[15] = 1;
	}

	void createRotateMatZ(float *R, float radians)
	{
        //set all values to zero
        for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
				R[i*4+j] = 0;
			}
		}

        R[0] = cos(radians);
        R[1] = sin(radians);
        R[4] = -sin(radians);
        R[5] = cos(radians);
        R[10] = 1;
        R[15] = 1;
	}

	void multMat(float *C, const float *A, const float *B)
	{
		float c = 0;
		for(int k = 0; k < 4; ++k) {
      // Process kth column of C
			for(int i = 0; i < 4; ++i) {
         // Process ith row of C.
         // The (i,k)th element of C is the dot product
         // of the ith row of A and kth col of B.
				c = 0;
         //vector dot
				for(int j = 0; j < 4; ++j) {
                    c += (A[i+4*j]*B[k*4+j]);
				}
                C[i+4*k] = c;
			}
		}
	}

	void createPerspectiveMat(float *m, float fovy, float aspect, float zNear, float zFar)
	{
   // http://www-01.ibm.com/support/knowledgecenter/ssw_aix_61/com.ibm.aix.opengl/doc/openglrf/gluPerspective.htm%23b5c8872587rree
		float f = 1.0f/glm::tan(0.5f*fovy);
		m[ 0] = f/aspect;
		m[ 1] = 0.0f;
		m[ 2] = 0.0f;
		m[ 3] = 0.0f;
		m[ 4] = 0;
		m[ 5] = f;
		m[ 6] = 0.0f;
		m[ 7] = 0.0f;
		m[ 8] = 0.0f;
		m[ 9] = 0.0f;
		m[10] = (zFar + zNear)/(zNear - zFar);
		m[11] = -1.0f;
		m[12] = 0.0f;
		m[13] = 0.0f;
		m[14] = 2.0f*zFar*zNear/(zNear - zFar);
		m[15] = 0.0f;
	}


	void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
	{
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		{
			glfwSetWindowShouldClose(window, GL_TRUE);
		}
		if (key == GLFW_KEY_Z && action == GLFW_PRESS) {
			glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
		}
		if (key == GLFW_KEY_Z && action == GLFW_RELEASE) {
			glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
		}
        if (key == GLFW_KEY_A && action == GLFW_PRESS) {
			sceneRotate += 0.3;
		}
		if (key == GLFW_KEY_D && action == GLFW_PRESS) {
			sceneRotate -= 0.3;
		}
	}

	void mouseCallback(GLFWwindow *window, int button, int action, int mods)
	{
		double posX, posY;

		if (action == GLFW_PRESS)
		{
			glfwGetCursorPos(window, &posX, &posY);
			cout << "Pos X " << posX <<  " Pos Y " << posY << endl;
		}
	}

	void resizeCallback(GLFWwindow *window, int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void init(const std::string& resourceDirectory)
	{
		GLSL::checkVersion();

		// Set background color.
		glClearColor(.12f, .34f, .56f, 1.0f);
		// Enable z-buffer test.
		glEnable(GL_DEPTH_TEST);

		// Initialize the GLSL program.
		prog = make_shared<Program>();
		prog->setVerbose(true);
		prog->setShaderNames(resourceDirectory + "/simple_vert.glsl", resourceDirectory + "/simple_frag.glsl");
		prog->init();
		prog->addUniform("P");
		prog->addUniform("V");
		prog->addUniform("M");
		prog->addAttribute("vertPos");
		prog->addAttribute("vertNor");

        // Initialize the GLSL program.
		solidColorProg = make_shared<Program>();
		solidColorProg->setVerbose(true);
		solidColorProg->setShaderNames(resourceDirectory + "/simple_vert.glsl", resourceDirectory + "/solid_frag.glsl");
		solidColorProg->init();
		solidColorProg->addUniform("P");
		solidColorProg->addUniform("V");
		solidColorProg->addUniform("M");
		solidColorProg->addUniform("solidColor");
		solidColorProg->addAttribute("vertPos");
		solidColorProg->addAttribute("vertNor");
	}

    void findMin(float x, float y, float z, string mesh) {
        if (mesh == "santa") {
            if (x < santaMin.x)
                santaMin.x = x;
            if (y < santaMin.y)
                santaMin.y = y;
            if (z < santaMin.z)
                santaMin.z = z;
        } else if (mesh == "sleigh") {
            if (x < sleighMin.x)
                sleighMin.x = x;
            if (y < sleighMin.y)
                sleighMin.y = y;
            if (z < sleighMin.z)
                sleighMin.z = z;
        } else if (mesh == "house") {
            if (x < houseMin.x)
                houseMin.x = x;
            if (y < houseMin.y)
                houseMin.y = y;
            if (z < houseMin.z)
                houseMin.z = z;
        } else {
            if (x < carMin.x)
                carMin.x = x;
            if (y < carMin.y)
                carMin.y = y;
            if (z < carMin.z)
                carMin.z = z;
        }
    }

    void findMax(float x, float y, float z, string mesh) {
        if (mesh == "santa") {
            if (x > santaMax.x)
                santaMax.x = x;
            if (y > santaMax.y)
                santaMax.y = y;
            if (z > santaMax.z)
                santaMax.z = z;
        } else if (mesh == "sleigh") {
            if (x > sleighMax.x)
                sleighMax.x = x;
            if (y > sleighMax.y)
                sleighMax.y = y;
            if (z > sleighMax.z)
                sleighMax.z = z;
        } else if (mesh == "house") {
            if (x > houseMax.x)
                houseMax.x = x;
            if (y > houseMax.y)
                houseMax.y = y;
            if (z > houseMax.z)
                houseMax.z = z;
        } else {
            if (x > carMax.x)
                carMax.x = x;
            if (y > carMax.y)
                carMax.y = y;
            if (z > carMax.z)
                carMax.z = z;
        }
    }

	void initGeom(const std::string& resourceDirectory)
	{

		//EXAMPLE set up to read one shape from one obj file - convert to read several
		// Initialize mesh
		// Load geometry
 		// Some obj files contain material information.We'll ignore them for this assignment.
		vector<tinyobj::shape_t> TOshapes;
		vector<tinyobj::material_t> objMaterials;
		string errStr;
		//load in the mesh and make the shape(s)
        bool rc = tinyobj::LoadObj(TOshapes, objMaterials, errStr, (resourceDirectory + "/cube.obj").c_str());
		if (!rc) {
			cerr << errStr << endl;
		} else {
            floorMesh = make_shared<Shape>(false);
            floorMesh->createShape(TOshapes[0]);
            floorMesh->measure();
            floorMesh->init();
		}

        rc = tinyobj::LoadObj(TOshapes, objMaterials, errStr, (resourceDirectory + "/streetlamp.obj").c_str());
		if (!rc) {
			cerr << errStr << endl;
		} else {
            lampMesh = make_shared<Shape>(false);
            lampMesh->createShape(TOshapes[0]);
            lampMesh->measure();
            lampMesh->init();
		}

        rc = tinyobj::LoadObj(TOshapes, objMaterials, errStr, (resourceDirectory + "/xmas.obj").c_str());
		if (!rc) {
			cerr << errStr << endl;
		} else {
            treeMesh = make_shared<Shape>(false);
            treeMesh->createShape(TOshapes[0]);
            treeMesh->measure();
            treeMesh->init();
		}

        rc = tinyobj::LoadObj(TOshapes, objMaterials, errStr, (resourceDirectory + "/sleigh.obj").c_str());
        if (!rc) {
			cerr << errStr << endl;
		} else {
            for (int i = 0; i < TOshapes.size(); i++) {
                shared_ptr<Shape> tmp = make_shared<Shape>(false);
                tmp->createShape(TOshapes[i]);
                tmp->measure();
                tmp->init();

                findMin(tmp->min.x, tmp->min.y, tmp->min.z, "sleigh");
                findMax(tmp->max.x, tmp->max.y, tmp->max.z, "sleigh");

                sleighMesh.push_back(tmp);
            }
		}

        rc = tinyobj::LoadObj(TOshapes, objMaterials, errStr, (resourceDirectory + "/santa.obj").c_str());
        if (!rc) {
			cerr << errStr << endl;
		} else {
            for (int i = 0; i < TOshapes.size(); i++) {
                shared_ptr<Shape> tmp = make_shared<Shape>(false);
                tmp->createShape(TOshapes[i]);
                tmp->measure();
                tmp->init();

                findMin(tmp->min.x, tmp->min.y, tmp->min.z, "santa");
                findMax(tmp->max.x, tmp->max.y, tmp->max.z, "santa");

                santaMesh.push_back(tmp);
            }
		}

        rc = tinyobj::LoadObj(TOshapes, objMaterials, errStr, (resourceDirectory + "/smallHouse.obj").c_str());
        if (!rc) {
			cerr << errStr << endl;
		} else {
            for (int i = 0; i < TOshapes.size(); i++) {
                shared_ptr<Shape> tmp = make_shared<Shape>(false);
                tmp->createShape(TOshapes[i]);
                tmp->measure();
                tmp->init();

                findMin(tmp->min.x, tmp->min.y, tmp->min.z, "house");
                findMax(tmp->max.x, tmp->max.y, tmp->max.z, "house");
                
                houseMesh.push_back(tmp);
            }
		}


        rc = tinyobj::LoadObj(TOshapes, objMaterials, errStr, (resourceDirectory + "/car.obj").c_str());
		if (!rc) {
			cerr << errStr << endl;
		} else {
            for (int i = 0; i < TOshapes.size(); i++) {
                shared_ptr<Shape> tmp = make_shared<Shape>(false);
                tmp->createShape(TOshapes[i]);
                tmp->measure();
                tmp->init();

                findMin(tmp->min.x, tmp->min.y, tmp->min.z, "car");
                findMax(tmp->max.x, tmp->max.y, tmp->max.z, "car");

                carMesh.push_back(tmp);
            }
		}
	}

    float findCenter(float min, float max) {
        float dist = (max-min)/2;
        return max-dist;
    }

    void drawTrees(float M[], float V[], float P[]) {
        float treeTrans[16] = {0};
        float treeScale[16] = {0};
        float treeRotate[16] = {0};
        float intermediate[16] = {0};

        float zVals[10] = {30, 15, 0, -15, -30, 30, 15, 0, -15, -30};
        float xVals[10] = {15, 15, 15, 15, 15, -15, -15,-15, -15, -15};

        createIdentityMat(intermediate);
        createScaleMat(treeScale, 0.5, 0.5, 0.5);
        createRotateMatY(treeRotate, 1.5);
        multMat(intermediate, treeRotate, treeScale);

        for (int j=0; j < 10; j++) {
            createTranslateMat(treeTrans, xVals[j], 0, zVals[j]);
            multMat(M, treeTrans, intermediate);
            glUniformMatrix4fv(prog->getUniform("M"), 1, GL_FALSE, M);
            treeMesh->draw(prog);
        }
    }

    void drawLamps(float M[], float V[], float P[]) {
        float lampTrans[16] = {0};
        float lampScale[16] = {0};
        float lampRotate[16] = {0};
        float intermediate[16] = {0};

        float zVals[4] = {22.5, 7.5, -7.5, -22.5};

        createScaleMat(lampScale, 3, 3, 3);
        createIdentityMat(intermediate);
        createRotateMatY(lampRotate, 3.14159);

        for (int l=0; l < 4; l++) {
            createTranslateMat(lampTrans, 15, 0, zVals[l]);
            multMat(M, lampTrans, lampScale);
            glUniformMatrix4fv(prog->getUniform("M"), 1, GL_FALSE, M);
            lampMesh->draw(prog);
        }

        for (int r=0; r < 4; r++) {
            multMat(intermediate, lampRotate, lampScale);
            createTranslateMat(lampTrans, -15, 0, zVals[r]);
            multMat(M, lampTrans, intermediate);
            glUniformMatrix4fv(prog->getUniform("M"), 1, GL_FALSE, M);
            lampMesh->draw(prog);
        }
    }

    void drawHouse(float M[], float V[], float P[]) {
        float houseTrans[16] = {0};
        float houseScale[16] = {0};
        float houseRotate[16] = {0};
        float intermediate[16] = {0};
        float initialTrans[16] = {0};

        float center = findCenter(houseMin.z, houseMax.z);
        createTranslateMat(initialTrans, 0, 0, -center);

        createScaleMat(houseScale, 0.07, 0.07, 0.07);
        createRotateMatY(houseRotate, 1.5);
        createTranslateMat(houseTrans, 0, 0, -23);
        multMat(M, houseScale, initialTrans);
        multMat(intermediate, houseRotate, M);
        multMat(M, houseTrans, intermediate);

		glUniformMatrix4fv(prog->getUniform("M"), 1, GL_FALSE, M);
        for (int i=0; i < houseMesh.size(); i++) {
            houseMesh[i]->draw(prog);
        }
    }

    void drawCar(float M[], float V[], float P[]) {
        float carTrans[16] = {0};
        float carScale[16] = {0};
        float carFirstTrans[16] = {0};
        float intermediate[16] = {0};
        driveTheta = 3*sin(glfwGetTime());

        createScaleMat(carScale, 0.4, 0.4, 0.4);
        createTranslateMat(carFirstTrans, -5, 0.25, 8);
        createTranslateMat(carTrans, 0, 0, driveTheta);
        multMat(intermediate, carFirstTrans, carScale);
        multMat(M, carTrans, intermediate);

		glUniformMatrix4fv(prog->getUniform("M"), 1, GL_FALSE, M);
        for (int i=0; i < carMesh.size(); i++) {
            carMesh[i]->draw(prog);
        }
    }

    void drawDecorations(float M[], float V[], float P[]) {
        float santaTrans[16] = {0};
        float santaScale[16] = {0};
        float intermediate[16] = {0};
        float initialTrans[16] = {0};

        float center = findCenter(santaMin.z, santaMax.z);
        createTranslateMat(initialTrans, 0, 0, -center);

        createScaleMat(santaScale, 0.0125, 0.0125, 0.0125);
        createTranslateMat(santaTrans, -7, 0, -15);
        multMat(intermediate, santaScale, initialTrans);
        multMat(M, santaTrans, intermediate);

		glUniformMatrix4fv(prog->getUniform("M"), 1, GL_FALSE, M);
        for (int i=0; i < santaMesh.size(); i++) {
            santaMesh[i]->draw(prog);
        }

        float sleighTrans[16] = {0};
        float sleighScale[16] = {0};
        float sleighRotate[16] = {0};

        center = findCenter(sleighMin.z, sleighMax.z);
        createTranslateMat(initialTrans, 0, 0, -center);

        createScaleMat(sleighScale, 1.25, 1.25, 1.25);
        createRotateMatY(sleighRotate, 1.5);
        createTranslateMat(sleighTrans, 0, 9, -25.3);
        multMat(M, sleighScale, initialTrans);
        multMat(intermediate, sleighRotate, M);
        multMat(M, sleighTrans, intermediate);

		glUniformMatrix4fv(prog->getUniform("M"), 1, GL_FALSE, M);
        for (int i=0; i < sleighMesh.size(); i++) {
            sleighMesh[i]->draw(prog);
        }
    }

	void render()
	{
		//local modelview matrix use this for lab 4
		float M[16] = {0};
		float V[16] = {0};
		float P[16] = {0};

		// Get current frame buffer size.
		int width, height;
		glfwGetFramebufferSize(windowManager->getHandle(), &width, &height);
		glViewport(0, 0, width, height);

		// Clear framebuffer.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//Use the local matrices for lab 4
		float aspect = width/(float)height;
		createPerspectiveMat(P, 70.0f, aspect, 0.1, 100.0f);	
		createIdentityMat(M);
        float globalTrans[16] = {0};
        float globalRotate[16] = {0};
        createTranslateMat(globalTrans, 0, -4, -25);
        createRotateMatY(globalRotate, sceneRotate);
        multMat(V, globalTrans, globalRotate);

        // Draw floor
        float floorTrans[16] = {0};
        float floorScale[16] = {0};

        createScaleMat(floorScale, 50, 0.5, 70);
        createTranslateMat(floorTrans, 0, 0, 0);
        multMat(M, floorTrans, floorScale);


        solidColorProg->bind();
		glUniformMatrix4fv(solidColorProg->getUniform("P"), 1, GL_FALSE, P);
		glUniformMatrix4fv(solidColorProg->getUniform("V"), 1, GL_FALSE, V);
        glUniformMatrix4fv(prog->getUniform("M"), 1, GL_FALSE, M);
		glUniform3f(solidColorProg->getUniform("solidColor"), 0.3, 0.7, 0.4);
        floorMesh->draw(prog);
		solidColorProg->unbind();

        prog->bind();
        glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, P);
		glUniformMatrix4fv(prog->getUniform("V"), 1, GL_FALSE, V);
        // Draw house
        drawHouse(M, V, P);

        // Draw santa and sleigh
        drawDecorations(M, V, P);

        // Draw car
        drawCar(M, V, P);

        // Draw trees
        drawTrees(M, V, P);

        // Draw streetlamps
        drawLamps(M, V, P);
        prog->unbind();
	}
};

int main(int argc, char *argv[])
{
	// Where the resources are loaded from
	std::string resourceDir = "../resources";

	if (argc >= 2)
	{
		resourceDir = argv[1];
	}

	Application *application = new Application();

	// Your main will always include a similar set up to establish your window
	// and GL context, etc.

	WindowManager *windowManager = new WindowManager();
	windowManager->init(640, 480);
	windowManager->setEventCallbacks(application);
	application->windowManager = windowManager;

	// This is the code that will likely change program to program as you
	// may need to initialize or set up different data and state

	application->init(resourceDir);
	application->initGeom(resourceDir);

	// Loop until the user closes the window.
	while (! glfwWindowShouldClose(windowManager->getHandle()))
	{
		// Render scene.
		application->render();

		// Swap front and back buffers.
		glfwSwapBuffers(windowManager->getHandle());
		// Poll for and process events.
		glfwPollEvents();
	}

	// Quit program.
	windowManager->shutdown();
	return 0;
}
