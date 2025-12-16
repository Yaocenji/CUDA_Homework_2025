#pragma once

#include <vector>
#include <iostream>
#include <omp.h>

#include <cuda_runtime.h>

#include "src/crigid.h"


#define CUDA_CHECK_ERROR(err) {\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

// 检测模式
enum CheckMode{
    CHECK_MODE_BRUTE_FORCE = 0,
    CHECK_MODE_BVH = 1,
    CHECK_MODE_TEST = 2
};

// BVH数据结构
struct LBVHNode {
    // 包围盒 (使用 double)
    vec3f min_box;
    vec3f max_box;

    // 拓扑关系
    int left_child;  // 索引
    int right_child; // 索引
    int parent;      // 索引

    // 用于自底向上 Refit 的原子计数器
    int atom_flag;
};

/// <summary>
/// HOST 函数：使用CUDA计算点集的 AABB
/// </summary>
/// <param name="points">in 点集指针 cuda</param>
/// <param name="num_points">in 点的数量</param>
/// <param name="min">out 输出的AABB最小 cuda</param>
/// <param name="max">out 输出的AABB最大 cuda</param>
void compute_global_aabb_double_cuda(vec3f* points, int num_points, vec3f* fmin, vec3f* fmax);

/// <summary>
/// HOST 函数：计算点集的 AABB
/// </summary>
/// <param name="points">in 点集指针</param>
/// <param name="num_points">in 点的数量</param>
/// <param name="min">out 输出的AABB最小</param>
/// <param name="max">out 输出的AABB最大</param>
void compute_global_aabb_double_cpu(vec3f* points, int num_points, vec3f* fmin, vec3f* fmax);

/// <summary>
/// 暴力计算两组点集之间的距离
/// </summary>
/// <param name="points1">点集1</param>
/// <param name="points2">点集2</param>
/// <param name="distData">距离数组</param>
/// <param name="pointNumber1">点集1长度</param>
/// <param name="pointNumber2">点集2长度</param>
/// <returns></returns>
__global__ void BruteForce_CalculateDistance(vec3f* points1, vec3f* points2, double* distData, size_t pointNumber1, size_t pointNumber2);

REAL checkDistCuda(const kmesh* m1, const kmesh* m2, std::vector<id_pair>& rets, CheckMode mode, bool stepTimeRecord);