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
#include <GL/gl.h>

#include <stdio.h>
#include <string.h>

#include "crigid.h"
#include "box.h"

#include <set>
using namespace std;

extern std::vector<int> fset;
extern std::set<int> vset;
extern vec3f projDir;
extern REAL maxDist;

// for fopen
#pragma warning(disable: 4996)

inline vec3f update(vec3f &v1, vec3f &v2, vec3f &v3)
{
	vec3f s = (v2-v1);
	return s.cross(v3-v1);
}

inline vec3f
update(tri3f &tri, vec3f *vtxs)
{
	vec3f &v1 = vtxs[tri.id0()];
	vec3f &v2 = vtxs[tri.id1()];
	vec3f &v3 = vtxs[tri.id2()];

	return update(v1, v2, v3);
}

void kmesh::updateNrms()
{
	if (_fnrms == nullptr)
		_fnrms = new vec3f[_num_tri];

	if (_nrms == nullptr)
		_nrms = new vec3f[_num_vtx];

	for (unsigned int i = 0; i < _num_tri; i++) {
		vec3f n = ::update(_tris[i], _vtxs);
		n.normalize();
		_fnrms[i] = n;
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i] = vec3f::zero();

	for (unsigned int i=0; i<_num_tri; i++) {
		vec3f& n = _fnrms[i];
		_nrms[_tris[i].id0()] += n;
		_nrms[_tris[i].id1()] += n;
		_nrms[_tris[i].id2()] += n;
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i].normalize();
}

void kmesh::display(bool cyl, int level)
{
	if (_dl == -1)
		updateDL(cyl, level);

	glCallList(_dl);
	displayDynamic(cyl, level);
}

void kmesh::displayStatic(bool cyl, int level)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
#ifdef USE_DOUBLE
	glVertexPointer(3, GL_DOUBLE, sizeof(REAL) * 3, _vtxs);
	glNormalPointer(GL_DOUBLE, sizeof(REAL) * 3, _nrms);
#else
	glVertexPointer(3, GL_FLOAT, sizeof(REAL) * 3, _vtxs);
	glNormalPointer(GL_FLOAT, sizeof(REAL) * 3, _nrms);
#endif

	glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris);
	// �������ж��㣺_num_vtx Ϊ��������
	//glDrawArrays(GL_POINTS, 0, _num_vtx);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

}

void kmesh::displayDynamic(bool cyl, int level)
{
}

void kmesh::destroyDL()
{
	if (_dl != -1)
		glDeleteLists(_dl, 1);
}

void kmesh::updateDL(bool cyl, int level)
{
	if (_dl != -1)
		destroyDL();

	_dl = glGenLists(1);
	glNewList(_dl, GL_COMPILE);
	displayStatic(cyl, level);
	glEndList();
}

extern std::vector<vec3f> vtxset;

void drawOther()
{
}

void beginDraw(BOX& bx)
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	vec3f pt = bx.center();
	REAL len = bx.height() + bx.depth() + bx.width();
	REAL sc = 6.0 / len;

	//glRotatef(-90, 0, 0, 1);
	glScalef(sc, sc, sc);
	glTranslatef(-pt.x, -pt.y, -pt.z);
}

void endDraw()
{
	glPopMatrix();
}


void drawRigid(crigid* r, bool cyl, int level, vec3f& off)
{
	glPushMatrix();
	glTranslated(off.x, off.y, off.z);

	//glLoadIdentity();
	matrix3f R = r->getRot();
	vec3f T = r->getOffset();
	std::vector<GLdouble> Twc = { R(0, 0), R(1, 0), R(2, 0), 0.,
							  R(0, 1), R(1, 1), R(2, 1), 0.,
							  R(0, 2), R(1, 2), R(2, 2), 0.,
							  T[0], T[1], T[2], 1. };

	glMultMatrixd(Twc.data());
	r->getMesh()->display(cyl, level);
	glPopMatrix();
}

void drawCDPair(crigid* r0, crigid* r1, std::vector<id_pair>& pairs)
{
	if (pairs.size() == 0)
		return;

	transf& t0 = r0->getWorldTransform();
	transf& t1 = r1->getWorldTransform();
	kmesh* km0 = r0->getMesh();
	kmesh* km1 = r1->getMesh();

	glPointSize(5.0);
	glLineWidth(2.0);

	glDisable(GL_LIGHTING);
	{
		for (auto t : pairs) {
			unsigned int v0, v1;
			t.get(v0, v1);
			vec3f p0 = t0.getVertex(km0->getVtxs()[v0]);
			vec3f p1 = t1.getVertex(km1->getVtxs()[v1]);

			glColor3f(1, 0, 0);
			glBegin(GL_POINTS);

#ifdef USE_DOUBLE
			glVertex3dv(p0.v);
			glVertex3dv(p1.v);
#else
			glVertex3fv(p0.v);
			glVertex3fv(p1.v);
#endif
			glEnd();

			glColor3f(0, 1, 0);
			glBegin(GL_LINES);
#ifdef USE_DOUBLE
			glVertex3dv(p0.v);
			glVertex3dv(p1.v);
#else
			glVertex3fv(p0.v);
			glVertex3fv(p1.v);
#endif
			glEnd();
		}

		glEnable(GL_LIGHTING);
	}

	glPointSize(1.0);
}
