#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include "src/crigid.h"

__global__ void CalculateDistance(vec3f* points1, vec3f* points2, double* distData, size_t pointNumber1, size_t pointNumber2);

REAL checkDistCuda(const kmesh* m1, const kmesh* m2, std::vector<id_pair>& rets);