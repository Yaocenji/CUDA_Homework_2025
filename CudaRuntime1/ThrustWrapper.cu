#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/pair.h>
#include <cfloat> // DBL_MAX

// 使用 extern "C" 防止 C++ 名称修饰 (Name Mangling) 问题

// 排序
extern "C" void perform_thrust_sort(unsigned long long* d_keys, int* d_values, int num_points) {
    try {
        // 这里的代码用 C++17 编译，Thrust 很开心
        thrust::device_ptr<unsigned long long> t_keys(d_keys);
        thrust::device_ptr<int> t_values(d_values);
        thrust::sort_by_key(t_keys, t_keys + num_points, t_values);
    }
    catch (...) {}
}

// 查询最小值及其索引
extern "C" void perform_thrust_min_element(double* d_data, int num_points, double* out_min_val, int* out_idx) {
    try {
        // 1. 包装原生指针
        thrust::device_ptr<double> t_ptr(d_data);

        // 2. 找到指向最小元素的迭代器
        // min_element 返回的是一个迭代器，指向范围 [first, last) 中最小的元素
        auto result_iter = thrust::min_element(t_ptr, t_ptr + num_points);

        // 3. 计算索引
        // distance 计算两个迭代器之间的元素个数
        *out_idx = (int)thrust::distance(t_ptr, result_iter);

        // 4. 获取值
        // 对 device_ptr 的迭代器进行解引用 (*)，Thrust 会自动执行 Device 到 Host 的拷贝
        *out_min_val = *result_iter;
    }
    catch (...) {
        // 简单的错误处理
        *out_min_val = -1.0;
        *out_idx = -1;
    }
}



// 我们需要在 Wrapper 里定义一个简单的结构来承载 vec3f 的数据
struct WrapperVec3 {
    double x, y, z;
};

// 定义包围盒结构
struct WrapperAABB {
    WrapperVec3 min_p;
    WrapperVec3 max_p;
};

// Functor 1: 将点转换为包围盒
struct PointToAABB {
    __host__ __device__
        WrapperAABB operator()(const WrapperVec3& p) const {
        return { p, p }; // 一个点的包围盒就是它自己
    }
};

// Functor 2: 合并两个包围盒 (Reduction)
struct MergeAABB {
    __host__ __device__
        WrapperAABB operator()(const WrapperAABB& a, const WrapperAABB& b) const {
        WrapperVec3 new_min, new_max;

        new_min.x = fmin(a.min_p.x, b.min_p.x);
        new_min.y = fmin(a.min_p.y, b.min_p.y);
        new_min.z = fmin(a.min_p.z, b.min_p.z);

        new_max.x = fmax(a.max_p.x, b.max_p.x);
        new_max.y = fmax(a.max_p.y, b.max_p.y);
        new_max.z = fmax(a.max_p.z, b.max_p.z);

        return { new_min, new_max };
    }
};

// 封装函数
extern "C" void perform_thrust_aabb(double* d_points, int num_points, double* out_min, double* out_max) {
    try {
        // 1. 将 double* 强制转换为 WrapperVec3* (前提是 vec3f 内存布局就是 x,y,z)
        WrapperVec3* ptr = reinterpret_cast<WrapperVec3*>(d_points);
        thrust::device_ptr<WrapperVec3> t_points(ptr);

        // 2. 初始化值 (Min 为无穷大，Max 为负无穷大)
        WrapperAABB init_val;
        init_val.min_p = { DBL_MAX, DBL_MAX, DBL_MAX };
        init_val.max_p = { -DBL_MAX, -DBL_MAX, -DBL_MAX };

        // 3. 执行 transform_reduce
        WrapperAABB result = thrust::transform_reduce(
            t_points,
            t_points + num_points,
            PointToAABB(), // 变换：点 -> Box
            init_val,      // 初始值
            MergeAABB()    // 归约：Box + Box -> Box
        );

        // 4. 将结果拷回输出指针 (注意：这些指针指向的是 Device Memory 还是 Host Memory？)
        // 假设 out_min/out_max 是 Device 指针，我们需要 cudaMemcpy
        // 假设 out_min/out_max 是 Host 指针，直接赋值

        // 【重要】根据你之前的 compute_aabb_kernel_double，你是传 Device 指针进去的
        // 所以我们需要拷贝回 Device
        cudaMemcpy(out_min, &result.min_p, sizeof(WrapperVec3), cudaMemcpyHostToDevice);
        cudaMemcpy(out_max, &result.max_p, sizeof(WrapperVec3), cudaMemcpyHostToDevice);
    }
    catch (...) {}
}