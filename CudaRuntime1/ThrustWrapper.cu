#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

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