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

#include "mat3f.h"
#include "quaternion.h"

#include <cuda_runtime.h>

class transf {
public:
	vec3f _off;
	matrix3f _trf;

	__host__ __device__ transf() {
		_trf = matrix3f::identity();
	}

	__host__ __device__ transf(const vec3f& off) {
		_off = off;
		_trf = matrix3f::identity();
	}

	__host__ __device__ transf(const vec3f& off, const matrix3f& trf) {
		_off = off;
		_trf = trf;
	}

	__host__ __device__ transf(const matrix3f& trf, const vec3f& off) {
		_off = off;
		_trf = trf;
	}

	__host__ __device__ FORCEINLINE void setOrigin(const vec3f& org) {
		_off = org;
	}

	__host__ __device__ FORCEINLINE void setRotation(const quaternion& q) {
		_trf = matrix3f::rotation(q);
	}

	__host__ __device__ FORCEINLINE void setRotation(const matrix3f &r) {
		_trf = r;
	}

	/**@brief Return the basis matrix for the rotation */
	__host__ __device__ FORCEINLINE matrix3f& getBasis() { return _trf; }
	/**@brief Return the basis matrix3f for the rotation.	 */
	__host__ __device__ FORCEINLINE const matrix3f& getBasis()    const { return _trf; }
	/**@brief Return the origin vector translation */
	__host__ __device__ FORCEINLINE vec3f& getOrigin() { return _off; }
	/**@brief Return the origin vector translation */
	__host__ __device__ FORCEINLINE const vec3f& getOrigin()   const { return _off; }

	/**@brief Return a quaternion representing the rotation */
	__host__ __device__ FORCEINLINE quaternion getRotation() const {
		quaternion q;
		_trf.getRotation(q);
		return q;
	}

	/**@brief Set this transformation to the identity */
	__host__ __device__ FORCEINLINE void setIdentity()
	{
		_trf = matrix3f::identity();
		_off = vec3f::zero();
	}

	__host__ __device__ FORCEINLINE vec3f getVertex(const vec3f& v) const
	{
		return _trf * v + _off;
	}

	__host__ __device__ FORCEINLINE vec3f getVertexInv(const vec3f& v) const
	{
		vec3f vv = v - _off;
		//return _trf.getInverse() * vv;
		return _trf.getTranspose() * vv;
	}

	__host__ __device__ FORCEINLINE transf inverse() const
	{
		matrix3f inv = _trf.getTranspose();
		return transf(inv, inv * -_off);
	}


	__host__ __device__ FORCEINLINE transf operator*(const transf& t) const
	{
		return transf(_trf * t._trf, (*this)(t._off));
	}

	/**@brief Return the transform of the vector */
	__host__ __device__ FORCEINLINE vec3f operator()(const vec3f& x) const
	{
		return x.dot3(
			vec3f(_trf(0, 0), _trf(0, 1), _trf(0, 2)),
			vec3f(_trf(1, 0), _trf(1, 1), _trf(1, 2)),
			vec3f(_trf(2, 0), _trf(2, 1), _trf(2, 2))) + _off;
	}
};

