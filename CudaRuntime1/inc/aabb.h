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

#include "vec3f.h"
#include "transf.h"
#include <float.h>
#include <vector>

class aabb {
public:
	__host__ __device__ FORCEINLINE void init() {
		_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	}

	vec3f _min;
	vec3f _max;

	__host__ __device__ FORCEINLINE aabb() {
		init();
	}

	__host__ __device__ FORCEINLINE aabb(const vec3f &v) {
		for (int i = 0; i < 3; i++) {
			_min[i] = v[i];
			_max[i] = v[i];
		}
	}

	__host__ __device__ FORCEINLINE aabb(const vec3f &a, const vec3f &b) {
		for (int i = 0; i < 3; i++) {
			_min[i] = fmin(a[i], b[i]);
			_max[i] = fmax(a[i], b[i]);
		}
	}

	__host__ __device__ FORCEINLINE aabb(const vec3f& a, const vec3f& b, const vec3f& c) {
		for (int i = 0; i < 3; i++) {
			_min[i] = fmin(a[i], fmin(b[i], c[i]));
			_max[i] = fmax(a[i], fmax(b[i], c[i]));
		}
		
	}

	__host__ __device__ FORCEINLINE bool overlaps(const aabb& b) const
	{
		if (_min[0] > b._max[0]) return false;
		if (_min[1] > b._max[1]) return false;
		if (_min[2] > b._max[2]) return false;

		if (_max[0] < b._min[0]) return false;
		if (_max[1] < b._min[1]) return false;
		if (_max[2] < b._min[2]) return false;

		return true;
	}

	__host__ __device__ FORCEINLINE bool overlaps(const aabb& b, REAL tol) const
	{
		aabb aa = *this;
		aabb bb = b;

		aa.enlarge(tol);
		bb.enlarge(tol);
		return aa.overlaps(bb);
	}

	__host__ __device__ FORCEINLINE bool overlaps(const aabb &b, aabb &ret) const
	{
		if (!overlaps(b))
			return false;

		ret._min = vec3f(
			fmax(_min[0], b._min[0]),
			fmax(_min[1], b._min[1]),
			fmax(_min[2], b._min[2]));

		ret._max = vec3f(
			fmin(_max[0], b._max[0]),
			fmin(_max[1], b._max[1]),
			fmin(_max[2], b._max[2]));

		return true;
	}

	__host__ __device__ FORCEINLINE bool inside(const vec3f &p) const
	{
		if (p[0] < _min[0] || p[0] > _max[0]) return false;
		if (p[1] < _min[1] || p[1] > _max[1]) return false;
		if (p[2] < _min[2] || p[2] > _max[2]) return false;

		return true;
	}

	__host__ __device__ FORCEINLINE aabb &operator += (const vec3f &p)
	{
		vmin(_min, p);
		vmax(_max, p);
		return *this;
	}

	__host__ __device__ FORCEINLINE aabb &operator += (const aabb &b)
	{
		vmin(_min, b._min);
		vmax(_max, b._max);
		return *this;
	}

	__host__ __device__ FORCEINLINE aabb operator + (const aabb &v) const
	{
		aabb rt(*this); return rt += v;
	}

	__host__ __device__ FORCEINLINE REAL width()  const { return _max[0] - _min[0]; }
	__host__ __device__ FORCEINLINE REAL height() const { return _max[1] - _min[1]; }
	__host__ __device__ FORCEINLINE REAL depth()  const { return _max[2] - _min[2]; }
	__host__ __device__ FORCEINLINE vec3f center() const { return (_min + _max)*REAL(0.5); }
	__host__ __device__ FORCEINLINE REAL volume() const { return width()*height()*depth(); }


	__host__ __device__ FORCEINLINE bool empty() const {
		return _max[0] < _min[0];
	}

	__host__ __device__ FORCEINLINE void enlarge(REAL thickness) {
		_max += vec3f(thickness, thickness, thickness);
		_min -= vec3f(thickness, thickness, thickness);
	}

	__host__ __device__ FORCEINLINE const vec3f &getMax() const { return _max; }
	__host__ __device__ FORCEINLINE const vec3f &getMin() const { return _min; }
	__host__ __device__ FORCEINLINE void setMax(vec3f& v) { _max = v; }
	__host__ __device__ FORCEINLINE void setMin(vec3f& v) { _min = v; }

	__host__ __device__ void getCorners(std::vector<vec3f> &crns) {
		crns.push_back(_max);
		crns.push_back(vec3f(_max.x, _max.y, _min.z));
		crns.push_back(vec3f(_max.x, _min.y, _min.z));
		crns.push_back(vec3f(_max.x, _min.y, _max.z));
		crns.push_back(_min);
		crns.push_back(vec3f(_min.x, _max.y, _min.z));
		crns.push_back(vec3f(_min.x, _max.y, _max.z));
		crns.push_back(vec3f(_min.x, _min.y, _max.z));
	}

	//! Apply a transform to an AABB
	__host__ __device__ FORCEINLINE void applyTransform(const transf &trans)
	{
		vec3f c = center();
		vec3f extends = _max - c;
		// Compute new center
		c = trans(c);

		vec3f textends = extends.dot3(trans.getBasis().getRow(0).absolute(),
			trans.getBasis().getRow(1).absolute(),
			trans.getBasis().getRow(2).absolute());

		_min = c - textends;
		_max = c + textends;
	}

	__host__ __device__ FORCEINLINE aabb transformed(const transf& trans) const
	{	
		aabb result;
		result.init();
		vec3f c = center();
		vec3f extends = _max - c;
		// Compute new center
		c = trans(c);

		vec3f textends = extends.dot3(trans.getBasis().getRow(0).absolute(),
			trans.getBasis().getRow(1).absolute(),
			trans.getBasis().getRow(2).absolute());

		result._min = c - textends;
		result._max = c + textends;
		return result;
	}

	//! Gets the extend and center
	__host__ __device__ FORCEINLINE void getCenterExtend(vec3f & center, vec3f & extend)  const
	{
		center = (_min + _max) * 0.5f;
		extend = _max - center;
	}

	__host__ __device__ int longestAxis() const {
		vec3f diff = _max - _min;
		if (diff.x > diff.y && diff.x > diff.z) {
			return 0;
		} else if (diff.y > diff.z) {
			return 1;
		} else {
			return 2;
		}
	}

	FORCEINLINE  bool intersectTriangleCPU(const vec3f& v0, const vec3f& v1, const vec3f& v2) const {
		vec3f boxCenter = (_min + _max) * 0.5f; // Center point of the AABB
		vec3f boxHalfSize = _max - boxCenter;    // Half size of the AABB
		vec3f triCenter = (v0 + v1 + v2) / 3.0f; // Center point of the triangle
		vec3f triHalfSize = maxCPU(maxCPU(v0 - triCenter, v1 - triCenter), v2 - triCenter); // Half size of the triangle

		// Test the vector from box center to triangle center
		vec3f d = triCenter - boxCenter;
		if (fabsf(d.x) > (boxHalfSize.x + triHalfSize.x)) return false;
		if (fabsf(d.y) > (boxHalfSize.y + triHalfSize.y)) return false;
		if (fabsf(d.z) > (boxHalfSize.z + triHalfSize.z)) return false;

		// Test the triangle normal as a separating axis
		vec3f e0 = v1 - v0;
		vec3f e1 = v2 - v1;
		vec3f e2 = v0 - v2;
		vec3f triNormal = crossCPU(e0, e1);
		if (!axisTestCPU(triNormal, v0, v1, v2, boxCenter, boxHalfSize)) return false;

		// Test the 3 face normals of the box (AABB's axes) as separating axes
		vec3f boxAxes[3] = { vec3f(1.0f, 0.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f), vec3f(0.0f, 0.0f, 1.0f) };
		for (int i = 0; i < 3; ++i) {
			if (!axisTestCPU(boxAxes[i], v0, v1, v2, boxCenter, boxHalfSize)) return false;
		}

		// Test the cross products of the box's 3 face normals and the triangle's 3 edges as separating axes
		for (int i = 0; i < 3; ++i) {
			vec3f axis = crossCPU(boxAxes[i], e0);
			if (!axisTestCPU(axis, v0, v1, v2, boxCenter, boxHalfSize)) return false;
			axis = crossCPU(boxAxes[i], e1);
			if (!axisTestCPU(axis, v0, v1, v2, boxCenter, boxHalfSize)) return false;
			axis = crossCPU(boxAxes[i], e2);
			if (!axisTestCPU(axis, v0, v1, v2, boxCenter, boxHalfSize)) return false;
		}

		// If none of the tests found a separating axis, then the AABB and the triangle intersect
		return true;
	}

	FORCEINLINE __host__ __device__ bool intersectTriangleGPU(const vec3f& v0, const vec3f& v1, const vec3f& v2) const {
		vec3f boxCenter = (_min + _max) * 0.5f; // Center point of the AABB
		vec3f boxHalfSize = _max - boxCenter;    // Half size of the AABB
		vec3f triCenter = (v0 + v1 + v2) / 3.0f; // Center point of the triangle
		vec3f triHalfSize = maxGPU(maxGPU(v0 - triCenter, v1 - triCenter), v2 - triCenter); // Half size of the triangle

		// Test the vector from box center to triangle center
		vec3f d = triCenter - boxCenter;
		if (fabsf(d.x) > (boxHalfSize.x + triHalfSize.x)) return false;
		if (fabsf(d.y) > (boxHalfSize.y + triHalfSize.y)) return false;
		if (fabsf(d.z) > (boxHalfSize.z + triHalfSize.z)) return false;

		// Test the triangle normal as a separating axis
		vec3f e0 = v1 - v0;
		vec3f e1 = v2 - v1;
		vec3f e2 = v0 - v2;
		vec3f triNormal = crossGPU(e0, e1);
		if (!axisTestGPU(triNormal, v0, v1, v2, boxCenter, boxHalfSize)) return false;

		// Test the 3 face normals of the box (AABB's axes) as separating axes
		vec3f boxAxes[3] = { vec3f(1.0f, 0.0f, 0.0f), vec3f(0.0f, 1.0f, 0.0f), vec3f(0.0f, 0.0f, 1.0f) };
		for (int i = 0; i < 3; ++i) {
			if (!axisTestGPU(boxAxes[i], v0, v1, v2, boxCenter, boxHalfSize)) return false;
		}

		// Test the cross products of the box's 3 face normals and the triangle's 3 edges as separating axes
		for (int i = 0; i < 3; ++i) {
			vec3f axis = crossGPU(boxAxes[i], e0);
			if (!axisTestGPU(axis, v0, v1, v2, boxCenter, boxHalfSize)) return false;
			axis = crossGPU(boxAxes[i], e1);
			if (!axisTestGPU(axis, v0, v1, v2, boxCenter, boxHalfSize)) return false;
			axis = crossGPU(boxAxes[i], e2);
			if (!axisTestGPU(axis, v0, v1, v2, boxCenter, boxHalfSize)) return false;
		}

		// If none of the tests found a separating axis, then the AABB and the triangle intersect
		return true;
	}

	__host__ __device__ void print(FILE *fp) {
		//fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}

	__host__ __device__ void visualize();

private:
	FORCEINLINE bool axisTestCPU(const vec3f& axis, const vec3f& v0, const vec3f& v1, const vec3f& v2, const vec3f& boxCenter, const vec3f& boxHalfSize) const {
		// Calculate the projection range of the triangle on the axis
		float p0 = dotCPU(axis, v0);
		float p1 = dotCPU(axis, v1);
		float p2 = dotCPU(axis, v2);
		float r = boxHalfSize.x * fabsf(axis.x) + boxHalfSize.y * fabsf(axis.y) + boxHalfSize.z * fabsf(axis.z);
		float minP = fminf(p0, fminf(p1, p2));
		float maxP = fmaxf(p0, fmaxf(p1, p2));
		// Calculate the projection range of the box on the axis
		float boxMin = dotCPU(axis, boxCenter) - r;
		float boxMax = dotCPU(axis, boxCenter) + r;
		// If the projections do not overlap, return false
		return maxP >= boxMin && minP <= boxMax;
	}

	FORCEINLINE __host__ __device__ bool axisTestGPU(const vec3f& axis, const vec3f& v0, const vec3f& v1, const vec3f& v2, const vec3f& boxCenter, const vec3f& boxHalfSize) const {
		// Calculate the projection range of the triangle on the axis
		float p0 = dotGPU(axis, v0);
		float p1 = dotGPU(axis, v1);
		float p2 = dotGPU(axis, v2);
		float r = boxHalfSize.x * fabsf(axis.x) + boxHalfSize.y * fabsf(axis.y) + boxHalfSize.z * fabsf(axis.z);
		float minP = fminf(p0, fminf(p1, p2));
		float maxP = fmaxf(p0, fmaxf(p1, p2));
		// Calculate the projection range of the box on the axis
		float boxMin = dotGPU(axis, boxCenter) - r;
		float boxMax = dotGPU(axis, boxCenter) + r;
		// If the projections do not overlap, return false
		return maxP >= boxMin && minP <= boxMax;
	}
};
