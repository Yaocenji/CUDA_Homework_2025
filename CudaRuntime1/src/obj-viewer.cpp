//**************************************************************************************
//  Copyright (C) 2022 - 2024, Min Tang (tang_m@zju.edu.cn)
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "Imgui/imgui.h"
#include "Imgui/imgui_impl_opengl3.h"

#include <GL/glh_glut.h>
#include <stdio.h>
#include <omp.h>

//#define PROF
using namespace glh;

bool b[256];
char* dataPath;
int win_w = 768, win_h = 768;
int stFrame = 0, pFrame = stFrame;
double totalQuery = 0;
static char fpsBuffer[512];

int steps = 0, num_streams = 1, preNumStreams = num_streams;
bool gpu = true, bvh = true, preGpu = gpu, preBvh = bvh;
bool isRotate = false, check = false, trackball = false;
bool show = true, collision = true, bunnyA = true, bunnyB = true;
bool light = true, verb = false;
const char* error_message = nullptr;
static int level = -1, drawMode = 0;
float lightpos[4] = { 13, 10.2, 3.2, 0 };
float clearColor[3] = { 1.0f, 1.0f, 1.0f };
GLfloat ambient = 0.f, diffuse = 1.f, specular = 0.f;
int modelType = 0, preModelType = -1;
int checkType = 2;
char* modelName;
char* fname1, * fname2;

glut_simple_mouse_interactor object;

// for sprintf
#pragma warning(disable: 4996)

extern void initModel(const char*);
extern void initModel(const char*, const char*);
extern void quitModel();
extern void drawModel(bool, bool, bool, bool, int);
extern bool dynamicModel(char*, bool, bool);
extern bool exportModel(const char*);
extern bool importModel(const char*);
extern void checkDistance(int mode);

// check for OpenGL errors
void checkGLError() {
	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR) {
		char msg[512];
		sprintf(msg, "error - %s\n", (char*)gluErrorString(error));
		printf(msg);
	}
}

void initLight() {
	GLfloat ambientLight[] = { ambient, ambient, ambient, 1.0f }; // ambient light
	GLfloat diffuseLight[] = { diffuse, diffuse, diffuse, 1.0f }; // diffuse light
	GLfloat specularLight[] = { specular, specular, specular, 1.0f }; // specular light

	glEnable(GL_LIGHTING);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
	glEnable(GL_LIGHT0);

	//GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
	//glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

	//glEnable(GL_COLOR_MATERIAL);
	//glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	//glMaterialfv(GL_FRONT, GL_SPECULAR, specularLight);
	//glMateriali(GL_FRONT, GL_SHININESS, 50);
}

void printLight() {
	printf("Light: %f, %f, %f, %f\n", lightpos[0], lightpos[1], lightpos[2], lightpos[3]);
}

void updateLight() {
	glLightfv(GL_LIGHT0, GL_POSITION, &lightpos[0]);
}

void initBackground() {
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, win_w, 0.0, win_h, -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glBegin(GL_QUADS);
		glColor3f(0.2, 0.4, 0.8);
		glVertex2f(0.0, 0.0);
		glVertex2f(win_w, 0.0);
		glColor3f(0.05, 0.1, 0.2);
		glVertex2f(win_w, win_h);
		glVertex2f(0, win_h);
	glEnd();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

void CalculateFrameRate() {
	static float framesPerSecond = 0.0f;
	static float lastTime = 0.0f;
	static bool first = true;
	if (first) {
		lastTime = GetTickCount() * 0.001f;
		first = false;
	}
	float currentTime = GetTickCount() * 0.001f;

	++framesPerSecond;
	float delta = currentTime - lastTime;
	if (delta > 1.0f) {
		lastTime = currentTime;
		sprintf(fpsBuffer, "FPS: %d", int(ceil(framesPerSecond)));
		framesPerSecond = 0;
	}
}

void draw() {
	glPushMatrix();
	drawModel(collision, show, bunnyA, bunnyB, level);
	glPopMatrix();
}

void CaptureScreen(int Width, int Height) {
#ifdef WIN32
	static int captures = 0;
	char filename[20];

	sprintf(filename, "Data/%04d.bmp", captures);
	captures++;

	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;

	char* image = new char[Width * Height * 3];
	FILE* file = fopen(filename, "wb");

	if (image != NULL) {
		if (file != NULL) {
			glReadPixels(0, 0, Width, Height, GL_BGR_EXT, GL_UNSIGNED_BYTE, image);

			memset(&bf, 0, sizeof(bf));
			memset(&bi, 0, sizeof(bi));

			bf.bfType = 'MB';
			bf.bfSize = sizeof(bf) + sizeof(bi) + Width * Height * 3;
			bf.bfOffBits = sizeof(bf) + sizeof(bi);
			bi.biSize = sizeof(bi);
			bi.biWidth = Width;
			bi.biHeight = Height;
			bi.biPlanes = 1;
			bi.biBitCount = 24;
			bi.biSizeImage = Width * Height * 3;

			fwrite(&bf, sizeof(bf), 1, file);
			fwrite(&bi, sizeof(bi), 1, file);
			fwrite(image, sizeof(unsigned char), Height * Width * 3, file);

			fclose(file);
		}
		delete[] image;
	}
#endif
}

void endCapture() {}

void key1() {
	dynamicModel(dataPath, false, false);
}

void key2() {
	if (preGpu != gpu || preBvh != bvh || preNumStreams != num_streams) {
		totalQuery = 0;
		steps = 0;
	}

	checkDistance(checkType);

	preGpu = gpu, preBvh = bvh, preNumStreams = num_streams;
	steps++;
}

void key3() {
	int steps = 350;
	for (int i = 0; i < steps; i++) key1();
}

void key4() {
	if (modelType == 0) {
		static int idx = 0;
		char buffer[512];
		sprintf(buffer, "../data/min-%05d.scn", idx++);
		importModel(buffer);
	}
}

void key5() {
	if (modelType == 0) {
		static int idx = 0;
		char buffer[512];
		sprintf(buffer, "../data/min-%05d.scn", idx++);
		//exportModel(buffer);
	}
}

void idle() {
	if (trackball) object.trackball.increment_rotation();

	if (isRotate) {
		key1();
		if (check) key2();
		else collision = false;
	}

	glutPostRedisplay();
}

void quit() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext();
	quitModel();
	exit(0);
}

void key(unsigned char k, int x, int y) {
	b[k] = !b[k];

	switch (k) {
		case 27:
		case 'q':
			quit();
			break;

		case 'x':
			{
				if (b['x'])
					printf("Starting screen capturing.\n");
				else
					printf("Ending screen capturing.\n");
				break;
			}

		case '1':
			key1();
			break;

		case '2':
			key2();
			break;

		case '3':
			key3();
			break;

		case '4':
			key4();
			break;

		case '5':
			key5();
			break;
	}

	object.keyboard(k, x, y);
	glutPostRedisplay();
}

void resize(int w, int h) {
	if (h == 0) h = 1;

	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 0.1, 500.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	object.reshape(w, h);

	win_w = w; win_h = h;
}

void mouse(int button, int state, int x, int y) {
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2((float)x, (float)y);
	io.MouseDown[button] = (state == GLUT_DOWN);
	switch (button) {
		case GLUT_LEFT_BUTTON:
			io.MouseDown[0] = state == GLUT_DOWN;
			break;
		case GLUT_RIGHT_BUTTON:
			io.MouseDown[1] = state == GLUT_DOWN;
			break;
		case GLUT_MIDDLE_BUTTON:
			io.MouseDown[2] = state == GLUT_DOWN;
			break;
	}

	if (!io.WantCaptureMouse) object.mouse(button, state, x, y);

	glutPostRedisplay();
}

void motion(int x, int y) {
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2(float(x), float(y));
	if (!io.WantCaptureMouse) object.motion(x, y);
}

void display() {
	static bool first = true;

	if (first) {
		totalQuery = 0;
		steps = 0;

		initModel(fname1, fname2);
		first = false;
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glShadeModel(GL_SMOOTH);
	glMatrixMode(GL_MODELVIEW);
	initBackground();
	object.apply_transform();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);

	// draw scene
	if (light) initLight();
	else {
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
	}
	if (drawMode) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	draw();

	// Start the ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui::NewFrame();
	ImGui::SetNextWindowSize(ImVec2(350, 180));
	ImGui::SetNextWindowPos(ImVec2(5, 5));
	ImGui::Begin("##main");
	{	
		ImGui::PushItemWidth(-1);
		//ImGui::Text("This is a GUI that controls the scene:");
		ImGui::Text("1 - Rotate models. 2 - Compute distance.");
		ImGui::Text("q/ESC - Quit. Left Btn - Obit.");
		ImGui::Text("Ctrl+Left Btn - Zoom. Shift+Left Btn - Pan.");
#if 0
		// Scene
		ImGui::AlignTextToFramePadding();
		ImGui::Text("Choose a scene:"); ImGui::SameLine();
		ImGui::RadioButton("Bunny", &modelType, 0); ImGui::SameLine();
		ImGui::RadioButton("Alien", &modelType, 1); ImGui::SameLine();
		ImGui::RadioButton("Wheeler", &modelType, 2);
#endif

		// GPU
		ImGui::RadioButton("CPU+Serial", &checkType, 0); ImGui::SameLine();
		ImGui::RadioButton("CPU+Parallel", &checkType, 1); ImGui::SameLine();
		ImGui::RadioButton("GPU", &checkType, 2);

#if 0
		ImGui::Checkbox("BVH", &bvh); ImGui::SameLine();
		ImGui::Checkbox("GPU", &gpu); ImGui::SameLine();
		ImGui::SetNextItemWidth(70);
		if (ImGui::InputInt("Number of streams (1-3)", &num_streams)) {
			if (num_streams <= 0 || num_streams > 3) {
				error_message = "Error: The number of streams must between 1 and 3.";
				num_streams = num_streams <= 0 ? 1 : 3;
			} else {
				error_message = nullptr;
			}
		}
		if (error_message != nullptr) {
			ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", error_message);
		}
#endif

		// Show
		ImGui::Checkbox("Show distance", &collision); ImGui::SameLine();
		//ImGui::Checkbox("Show body", &show); ImGui::SameLine();
		ImGui::Checkbox("Bunny A", &bunnyA); ImGui::SameLine();
		ImGui::Checkbox("Bunny B", &bunnyB);
#if 0
		// Light
		ImGui::Checkbox("Rotate", &isRotate); ImGui::SameLine();
		ImGui::Checkbox("Check", &check); ImGui::SameLine();
		ImGui::Checkbox("Trackball", &trackball); ImGui::SameLine();
		ImGui::Checkbox("Light", &light);
		ImGui::SliderFloat("Ambient Light", &ambient, 0.0f, 1.0f);
		ImGui::SliderFloat("Diffuse Light", &diffuse, 0.0f, 1.0f);
		ImGui::SliderFloat("Specular Light", &specular, 0.0f, 1.0f);
#endif

		// Drawing mode
		ImGui::AlignTextToFramePadding();
		ImGui::Text("Choose drawing mode:"); ImGui::SameLine();
		ImGui::RadioButton("Filling", &drawMode, 0); ImGui::SameLine();
		ImGui::RadioButton("Wireframe", &drawMode, 1);

		ImGui::Text("Average query time: %3.5f ms\n", steps == 0 ? 0 : totalQuery * 1000 / double(steps));
		ImGui::PopItemWidth();
	}
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glutSwapBuffers();
}

#ifdef PROF
int main(int argc, char** argv) {
	if (argc < 3) {
		printf("usage: %s model1.obj model2\n", argv[0]);
		return -1;
	}

	initModel(argv[1], argv[2]);

	// debug
	checkType = 2;

	totalQuery = 0;
	verb = false;
	int steps = 20;
	for (int i = 0; i < steps; i++) {
		printf("step %d of total %d.\n", i, steps);
		key1();
		key2();
	}

	printf("#average query time: %3.5f ms\n", totalQuery * 1000 / double(steps));
}
#else
int main(int argc, char** argv) {
	if (argc < 3) {
		printf("usage: %s model1.obj model2\n", argv[0]);
		return -1;
	}
	
	fname1 = argv[1];
	fname2 = argv[2];



	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_STENCIL);
	glutInitWindowSize(win_w, win_h);
	glutCreateWindow("GPU Distance Computation");
	glClearColor(clearColor[0], clearColor[1], clearColor[2], 1.0f);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.DisplaySize = ImVec2((float)win_w, (float)win_h);
	ImGui::StyleColorsDark();
	ImGui_ImplOpenGL3_Init("#version 130");

	object.configure_buttons(1);
	object.dolly.dolly[2] = -3;
	object.trackball.incr = rotationf(vec3f(1, 1, 0), 0.05);

	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutIdleFunc(idle);
	glutKeyboardFunc(key);
	glutReshapeFunc(resize);

	glutMainLoop();

	quit();
	return 0;
}
#endif
