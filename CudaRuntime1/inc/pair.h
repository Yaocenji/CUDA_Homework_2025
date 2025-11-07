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

class id_pair {
	unsigned int _id[2];

public:
	__host__ __device__ id_pair() {
		_id[0] = _id[1] = -1;
	}

	__host__ __device__ id_pair(unsigned int id1, unsigned int id2, bool sort)
	{
		_id[0] = id1;
		_id[1] = id2;

		if (sort && id1 > id2) {
			_id[0] = id2;
			_id[1] = id1;
		}
	}

	__host__ __device__ void get(unsigned int& id1, unsigned int& id2)
	{
		id1 = _id[0];
		id2 = _id[1];
	}

	__host__ __device__ bool operator < (const id_pair& other) const {
		if (_id[0] == other._id[0])
			return _id[1] < other._id[1];
		else
			return _id[0] < other._id[0];
	}

	__host__ __device__ bool operator == (const id_pair& other) const {
		return (_id[0] == other._id[0] && _id[1] == other._id[1]);
	}

};
