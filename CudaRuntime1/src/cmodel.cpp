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

#include "mat3f.h"
#include "box.h"
#include "crigid.h"
#include "pair.h"
#include <stdio.h>
#include <omp.h>
#include <mutex>
#include <iostream>

#include "../checkCuda.h"

using namespace std;

// mutex to lock critical region
std::mutex mylock;

#ifdef PROF
//#if defined(PROF) || defined(GPU)

class myTimer {
public:
	myTimer(const char* msgIn) {}
};


class myTimer2 {
public:
	myTimer2(const char* msgIn) {}

	void print() {}

	void inc(double delta) {}
};

#else

class myTimer {
	double t0;
	char msg[512];
public:
	myTimer(const char* msgIn) {
		t0 = omp_get_wtime();
		strcpy(msg, msgIn);
	}
	~myTimer() {
		double tdelta = omp_get_wtime() - t0;
		printf("%s: %2.5f s\n", msg, tdelta);
	}
};

class myTimer2 {
	double dt;
	char msg[512];
public:
	myTimer2(const char* msgIn) {
		dt = 0;
		strcpy(msg, msgIn);
	}

	void print() {
		printf("%s: %2.5f s\n", msg, dt);
	}

	void inc(double delta) {
		dt += delta;
	}
};

#endif

#define BUNNY_SCALE 1.f

#pragma warning(disable: 4996)

extern void drawSdfPair(crigid* r0, crigid* r1, std::vector<vec3f>& pairs);
extern void drawMinPair(crigid* r0, crigid* r1, std::vector<vec3f>& pairs);
extern void drawCDPair(crigid* r0, crigid* r1, std::vector<id_pair>& pairs);
extern void drawRigid(crigid*, bool cyl, int level, vec3f&);
extern void drawPlanes(bool);
extern float getLargestVelocityNorm(crigid* body1, crigid* body2);


BOX g_box;
BOX g_projBx;
REAL g_time = 0.0f;

extern bool verb;

vec3f projDir(0.0f, -1.0f, 0.0f);
REAL maxDist = 20.0;
static int sidx = 0;
double max_value = std::numeric_limits<double>::max();

class cscene {
	std::vector<kmesh*> _meshs;
	std::vector<crigid*> _rigids;

	//for GPU updating...
	std::vector< transf> _trfs;

public:
	~cscene() { clear(); }

	void clear() {
		for (auto r : _rigids)
			delete r;

		for (auto m : _meshs)
			delete m;

		_meshs.clear();
		_rigids.clear();
	}

	FORCEINLINE crigid* getRigid(int rid) {
		return (rid < 0) ? nullptr : _rigids[rid];
	}

	FORCEINLINE int getRigidID(crigid* r) {
		return r == nullptr ? -1 : r->getID();
	}

	FORCEINLINE void setID() {
		for (int i = 0; i < _rigids.size(); i++) {
			_rigids[i]->setID(i);
		}
	}

	void draw(int level, bool showCD, bool showBody, bool bodyA, bool bodyB) {
		if (showCD) {
			drawCDPair(_rigids[0], _rigids[1], cdPairs);
		}

		if (showBody) {
			vec3f dummy;
			if (bodyA)  drawRigid(_rigids[0], false, level, dummy);
			if (bodyB)  drawRigid(_rigids[1], false, level, dummy);
		}
	}

	void addMesh(kmesh* km) {
		_meshs.push_back(km);
	}

	void addRigid(crigid* rig) {
		_rigids.push_back(rig);
	}

	bool output(const char* fname) {
		FILE* fp = fopen(fname, "wt");
		if (fp == NULL)
			return false;

		fprintf(fp, "%zd\n", _rigids.size());
		for (int i = 0; i < _rigids.size(); i++) {
			transf& trf = _rigids[i]->getWorldTransform();
			vec3f& off = trf.getOrigin();
			quaternion q = trf.getRotation();
			fprintf(fp, "%lf, %lf, %lf\n", off.x, off.y, off.z);
			fprintf(fp, "%lf, %lf, %lf, %lf\n", q.x(), q.y(), q.z(), q.w());
		}
		fclose(fp);
		return true;
	}

	bool input(const char* fname) {
		FILE* fp = fopen(fname, "rt");
		if (fp == NULL)
			return false;

		int num = 0;
		char buffer[512];
		fgets(buffer, 512, fp);
		sscanf(buffer, "%d", &num);
		if (num != _rigids.size())
			return false;

		for (int i = 0; i < _rigids.size(); i++) {
			transf& trf = _rigids[i]->getWorldTransform();

			fgets(buffer, 512, fp);
			double x, y, z, w;
			sscanf(buffer, "%lf, %lf, %lf", &x, &y, &z);
			vec3f off(x, y, z);
			fgets(buffer, 512, fp);
			sscanf(buffer, "%lf, %lf, %lf, %lf", &x, &y, &z, &w);
			quaternion q(x, y, z, w);

			trf.setOrigin(off);
			trf.setRotation(q);
		}
		fclose(fp);
		return true;
	}

	//for collision detection
	std::vector<id_pair> cdPairs, cdPairs2;
}g_scene;

vec3f dPt0, dPt1, dPtw;

bool readobjfile(const char* path,
				 unsigned int& numVtx, unsigned int& numTri,
				 tri3f*& tris, vec3f*& vtxs, REAL scale, vec3f shift, bool swap_xyz, vec2f*& texs, tri3f*& ttris) {
	vector<tri3f> triset;
	vector<vec3f> vtxset;
	vector<vec2f> texset;
	vector<tri3f> ttriset;

	FILE* fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
			double x, y, z;
			sscanf(buf + 2, "%lf%lf%lf", &x, &y, &z);

			if (swap_xyz)
				vtxset.push_back(vec3f(z, x, y) * scale + shift);
			else
				vtxset.push_back(vec3f(x, y, z) * scale + shift);
		} else

			if (buf[0] == 'v' && buf[1] == 't') {
				double x, y;
				sscanf(buf + 3, "%lf%lf", &x, &y);

				texset.push_back(vec2f(x, y));
			} else
				if (buf[0] == 'f' && buf[1] == ' ') {
					int id0, id1, id2, id3 = 0;
					int tid0, tid1, tid2, tid3 = 0;
					bool quad = false;

					int count = sscanf(buf + 2, "%d/%d", &id0, &tid0);
					char* nxt = strchr(buf + 2, ' ');
					sscanf(nxt + 1, "%d/%d", &id1, &tid1);
					nxt = strchr(nxt + 1, ' ');
					sscanf(nxt + 1, "%d/%d", &id2, &tid2);

					nxt = strchr(nxt + 1, ' ');
					if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
						if (sscanf(nxt + 1, "%d/%d", &id3, &tid3))
							quad = true;
					}

					id0--, id1--, id2--, id3--;
					tid0--, tid1--, tid2--, tid3--;

					triset.push_back(tri3f(id0, id1, id2));
					if (count == 2) {
						ttriset.push_back(tri3f(tid0, tid1, tid2));
					}

					if (quad) {
						triset.push_back(tri3f(id0, id2, id3));
						if (count == 2)
							ttriset.push_back(tri3f(tid0, tid2, tid3));
					}
				}
	}
	fclose(fp);

	if (triset.size() == 0 || vtxset.size() == 0)
		return false;

	numVtx = vtxset.size();
	vtxs = new vec3f[numVtx];
	for (unsigned int i = 0; i < numVtx; i++)
		vtxs[i] = vtxset[i];

	int numTex = texset.size();
	if (numTex == 0)
		texs = NULL;
	else {
		texs = new vec2f[numTex];
		for (unsigned int i = 0; i < numTex; i++)
			texs[i] = texset[i];
	}

	numTri = triset.size();
	tris = new tri3f[numTri];
	for (unsigned int i = 0; i < numTri; i++)
		tris[i] = triset[i];

	int numTTri = ttriset.size();
	if (numTTri == 0)
		ttris = NULL;
	else {
		ttris = new tri3f[numTTri];
		for (unsigned int i = 0; i < numTTri; i++)
			ttris[i] = ttriset[i];
	}

	return true;
}

crigid* putMesh(kmesh* km, const vec3f& startPos, const vec3f& angle) {
	crigid* body = new crigid(km, startPos, 4.f);
	body->getWorldTransform().setOrigin(startPos);
	body->getWorldTransform().setRotation(quaternion(angle.x, angle.y, angle.z));
	return body;
}

kmesh* initBunny(const char* ofile) {
	unsigned int numVtx = 0, numTri = 0;
	vec3f* vtxs = NULL;
	tri3f* tris = NULL;
	vec2f* texs = NULL;
	tri3f* ttris = NULL;

	REAL scale = BUNNY_SCALE;
	vec3f shift(0, 0, 0);

	if (false == readobjfile(ofile, numVtx, numTri, tris, vtxs, scale, shift, false, texs, ttris)) {
		printf("loading %s failed...\n", ofile);
		// std::cout << "loading" << ofile << "failed...\n";
		exit(-1);
		// return nullptr;
	}

	kmesh* bunny = new kmesh(numVtx, numTri, tris, vtxs, false);

	g_scene.addMesh(bunny);

	return bunny;
}

void quitModel() {
	g_scene.clear();
}

void initModel(const char* cfile) {
	g_box.init();
	quitModel();

	kmesh* kmA = initBunny(cfile);
	kmesh* kmB = initBunny(cfile);

	crigid* rigA = putMesh(kmA, vec3f(), vec3f());
	g_box += rigA->bound();

	vec3f shift;
	if (cfile == "my-bunny.obj") {
		shift = vec3f(-1.25, 0, 0);
	} else if (cfile == "alien-animal.obj") {
		shift = vec3f(-3, 0, 0);
	} else if (cfile == "wheeler.obj") {
		shift = vec3f(-5, 0, 0);
	} else {
		shift = vec3f(-1, 0, 0);
	}
	crigid* rigB = putMesh(kmB, shift, vec3f());
	g_box += rigB->bound();

	g_scene.addRigid(rigA);
	g_scene.addRigid(rigB);
}

void initModel(const char* c1file, const char* c2file) {
	g_box.init();
	quitModel();

	kmesh* kmA = initBunny(c1file);
	kmesh* kmB = initBunny(c2file);

	crigid* rigA = putMesh(kmA, vec3f(), vec3f());
	g_box += rigA->bound();

	vec3f shift;
	if (c1file == "my-bunny.obj") {
		shift = vec3f(-1.25, 0, 0);
	} else if (c1file == "alien-animal.obj") {
		shift = vec3f(-3, 0, 0);
	} else if (c1file == "wheeler.obj") {
		shift = vec3f(-5, 0, 0);
	} else {
		shift = vec3f(-1, 0, 0);
	}
	crigid* rigB = putMesh(kmB, shift, vec3f());
	g_box += rigB->bound();

	g_scene.addRigid(rigA);
	g_scene.addRigid(rigB);
}

bool exportModel(const char* cfile) {
	return g_scene.output(cfile);
}

bool importModel(const char* cfile) {
	bool ret = g_scene.input(cfile);

	return ret;
}

extern void beginDraw(BOX&);
extern void endDraw();

void drawOther();

void drawBVH(int level) {
	NULL;
}

void setMat(int i, int id);

void drawModel(bool tri, bool pnt, bool bunnyA, bool bunnyB, int level) {
	if (!g_box.empty())
		beginDraw(g_box);

	drawOther();
	g_scene.draw(level, tri, pnt, bunnyA, bunnyB);

	drawBVH(level);

	if (!g_box.empty())
		endDraw();
}

extern double totalQuery;
extern int steps;

bool dynamicModel(char*, bool, bool) {
	static int st = 0;

	{
		crigid* body = g_scene.getRigid(0);
		transf& trf = body->getWorldTransform();
		vec3f axis(1, 1, 1);
		axis.normalize();
		matrix3f rot = matrix3f::rotation(axis, 0.1);
		matrix3f rold = trf.getBasis();
		matrix3f rnew = rold * rot;
		trf.setRotation(rnew);
	}

	{
		crigid* body = g_scene.getRigid(1);
		transf& trf = body->getWorldTransform();
		vec3f axis(-1, 1, 1);
		axis.normalize();
		matrix3f rot = matrix3f::rotation(axis, 0.1);
		matrix3f rold = trf.getBasis();
		matrix3f rnew = rold * rot;
		trf.setRotation(rnew);
	}

	return true;
}

REAL check(kmesh* m1, kmesh* m2, const transf& trfA, const transf& trfB, std::vector<id_pair>& pairs)
{
	const transf trfA2B = trfB.inverse() * trfA;
	return	m1->distNaive(m2, trfA2B, pairs);
}

REAL checkCuda(kmesh* m1, kmesh* m2, std::vector<id_pair>& pairs) {
	kmesh* realM1 = nullptr; 
	kmesh* realM2 = nullptr;
	if (m1->_num_vtx <= m2->_num_vtx) {
		realM1 = m1; realM2 = m2;
	}
	else {
		realM1 = m2; realM2 = m1;
	}
#ifdef PROF
	return checkDistCuda(m1, m2, g_scene.cdPairs, CHECK_MODE_BVH, false);
#else 
	return checkDistCuda(m1, m2, g_scene.cdPairs, CHECK_MODE_BVH, true);
#endif
}

void checkDistance(int mode)
{
	// std::cout << "Dist Mode: " << mode << std::endl;

	crigid* bodyA = g_scene.getRigid(0);
	crigid* bodyB = g_scene.getRigid(1);

	g_scene.cdPairs.clear();
	const transf& trfA = bodyA->getTrf();
	const transf& trfB = bodyB->getTrf();
	double tstart, tdelta;

	kmesh* m1 = bodyA->getMesh();
	kmesh* m2 = bodyB->getMesh();
	tstart = omp_get_wtime();

	
	float dist = 0;
	if (mode == 2)
		dist = checkCuda(m1, m2, g_scene.cdPairs);
	else
		dist = check(m1, m2, trfA, trfB, g_scene.cdPairs);

	tdelta = omp_get_wtime() - tstart;

	printf("MinDistance = %f (%zd pairs) at %2.5f s\n", sqrt(dist), g_scene.cdPairs.size(), tdelta);
	for (auto p : g_scene.cdPairs) {
		unsigned int id0, id1;
		p.get(id0, id1);
		vec3f v0 = m1->getVtxs()[id0];
		vec3f v1 = m2->getVtxs()[id1];
		vec3f wv0 = trfA.getVertex(v0);
		vec3f wv1 = trfB.getVertex(v1);
		printf("(%d, %d): (%f, %f, %f) - (%f, %f, %f) = %f\n", id0, id1, wv0.x, wv0.y, wv0.z, wv1.x, wv1.y, wv1.z, (wv0 - wv1).length());
	}

#if 0
	if (totalQuery > max_value - tdelta) {
		totalQuery = 0;
		steps = 0;
	}
#endif

	totalQuery += tdelta;
	return;
}

#include <omp.h>
#include <mutex>
std::mutex mtx;

REAL kmesh::distNaive(const kmesh* other, const transf& trf, std::vector<id_pair>& rets)
{
	REAL min_dist = 1e10;

#pragma omp parallel for
	for (int j = 0; j < _num_vtx; j++) {
		vec3f v1 = _vtxs[j];
		vec3f v11 = trf.getVertex(v1);

		for (int i = 0; i < other->_num_vtx; i++) {
			vec3f v0 = other->_vtxs[i];

			REAL d = (v11 - v0).squareLength();
			if (d < min_dist) {

				std::lock_guard<std::mutex> lock(mtx);
				min_dist = d;
				rets.clear();
				rets.push_back(id_pair(j, i, false));
			}
			else if (d == min_dist) {
				std::lock_guard<std::mutex> lock(mtx);
				rets.push_back(id_pair(j, i, false));
			}
		}
	}

	return min_dist;
}