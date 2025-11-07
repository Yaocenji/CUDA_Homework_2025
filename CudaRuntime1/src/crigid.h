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

#pragma once

#include <stdio.h>
#include "vec3f.h"
#include "mat3f.h"
#include "transf.h"

#include "tri.h"
#include "box.h"
#include "pair.h"

#include <set>
#include <vector>
#include <stack>

#include <cuda_runtime.h>
using namespace std;

class kmesh {
public:
	unsigned int _num_vtx;
	unsigned int _num_tri;
	tri3f* _tris;
	vec3f* _vtxs;

	vec3f* _fnrms;
	vec3f* _nrms;
	aabb* _bxs;
	aabb _bx;
	int _dl;

public:
	__host__ __device__ kmesh(unsigned int numVtx, unsigned int numTri, tri3f* tris, vec3f* vtxs, bool cyl) {
		_num_vtx = numVtx;
		_num_tri = numTri;
		_tris = tris;
		_vtxs = vtxs;

		_fnrms = nullptr;
		_nrms = nullptr;
		_bxs = nullptr;
		_dl = -1;

		updateNrms();
		updateDL(cyl, -1);
	}

	__host__ ~kmesh() {
		delete[]_tris;
		delete[]_vtxs;

		if (_fnrms != nullptr)
			delete[] _fnrms;
		if (_nrms != nullptr)
			delete[] _nrms;
		if (_bxs != nullptr)
			delete[] _bxs;
		destroyDL();
	}

	__host__ __device__ unsigned int getNbVertices() const { return _num_vtx; }
	__host__ __device__ unsigned int getNbFaces() const { return _num_tri; }
	__host__ __device__ vec3f* getVtxs() const { return _vtxs; }
	__host__ __device__ vec3f* getNrms() const { return _nrms; }
	__host__ __device__ vec3f* getFNrms() const { return _fnrms; }
	__host__ __device__ tri3f* getTris() const { return _tris; }

	// calc norms, and prepare for display ...
	__host__ __device__ void updateNrms();

	// really displaying ...
	__host__ __device__ void display(bool cyl, int);
	__host__ __device__ void displayStatic(bool cyl, int);
	__host__ __device__ void displayDynamic(bool cyl, int);

	// prepare for display
	__host__ __device__ void updateDL(bool cyl, int);
	// finish for display
	__host__ __device__ void destroyDL();

	__host__ __device__ void getTriangleVtxs(int fid, vec3f& v0, vec3f& v1, vec3f& v2) const
	{
		tri3f& f = _tris[fid];
		v0 = _vtxs[f.id0()];
		v1 = _vtxs[f.id1()];
		v2 = _vtxs[f.id2()];
	}

	__host__ __device__ BOX getBound(int i) const
	{
		return _bxs[i];
	}

	__host__ __device__ BOX bound() {
		return _bx;
	}



	REAL distNaive(const kmesh* other, const transf& trf, std::vector<id_pair>& rets);

private:
};

class crigid {
private:
	int _id;
	kmesh* _mesh;
	aabb _bx;
	transf _worldTrf;

	__host__ __device__ void updateBx() {
		_bx = _mesh->bound();
		_bx.applyTransform(_worldTrf);
	}

	//for bvh
	int _levelSt;
	int _levelNum;

public:
	__host__ __device__ crigid(kmesh* km, const transf& trf, float mass) {
		_mesh = km;
		updateBx();
		_levelSt = -1;
	}

	__host__ __device__ ~crigid() {
		NULL;
	}

	__host__ __device__ void setID(int i) { _id = i; }
	__host__ __device__ int getID() { return _id; }

	__host__ __device__ BOX bound() {
		return _bx;
	}

	__host__ __device__ kmesh* getMesh() {
		return _mesh;
	}

	__host__ __device__ const transf& getTrf() const {
		return _worldTrf;
	}

	__host__ __device__ const matrix3f& getRot() const
	{
		return _worldTrf._trf;
	}

	__host__ __device__ const vec3f& getOffset() const
	{
		return _worldTrf._off;
	}

	__host__ __device__ void updatePos(matrix3f& rt, vec3f offset = vec3f())
	{
		_worldTrf._trf = rt;
		_worldTrf._off = offset;

		updateBx();
	}

	__host__ __device__ int getVertexCount() {
		return _mesh->getNbVertices();
	}

	__host__ __device__ vec3f getVertex(int i) {
		vec3f& p = _mesh->getVtxs()[i];
		return _worldTrf.getVertex(p);
	}

	__host__ __device__ __forceinline transf& getWorldTransform() {
		return _worldTrf;
	}

	__host__ __device__ __forceinline void	setWorldTransform(const transf& worldTrans) {
		_worldTrf = worldTrans;
	}
};

