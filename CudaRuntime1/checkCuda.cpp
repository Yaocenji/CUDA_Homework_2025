#include "checkCuda.h"

// 实现 double 类型的 atomicMin
__device__ __forceinline__ double atomicMinDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		// 如果当前值已经比 val 小，则无需更新，直接中断
		if (__longlong_as_double(assumed) <= val) {
			return __longlong_as_double(old);
		}
		// 尝试更新：如果 address 处的值仍为 assumed，则写入 val (转换为 ull)
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
	} while (assumed != old);

	return __longlong_as_double(old);
}

// 实现 double 类型的 atomicMax
__device__ __forceinline__ double atomicMaxDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		// 如果当前值已经比 val 大，则无需更新
		if (__longlong_as_double(assumed) >= val) {
			return __longlong_as_double(old);
		}
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
	} while (assumed != old);

	return __longlong_as_double(old);
}



// 暴力计算点集之间的距离矩阵
__global__ void BruteForce_CalculateDistance(vec3f* points1, vec3f* points2, double* distData, size_t pointNumber1, size_t pointNumber2) {
	// 获取x，y坐标
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// 检查索引是否在范围内
	if (x < 0 || x >= pointNumber1 || y < 0 || y >= pointNumber2) {
		return;
	}

	// 计算点对之间的距离
	vec3f p1 = points1[x];
	vec3f p2 = points2[y];
	double dist = sqrt((p1.x - p2.x) * (p1.x - p2.x) +
		(p1.y - p2.y) * (p1.y - p2.y) +
		(p1.z - p2.z) * (p1.z - p2.z));

	// 将距离存储在distData中
	distData[y * pointNumber1 + x] = dist;
}



// 计算点集的 AABB (双精度版本)
__global__ void compute_aabb_kernel_double(const vec3f* points, int num_points, vec3f* global_min, vec3f* global_max) {
	// 1. 线程局部变量初始化 (使用 DBL_MAX)
	vec3f local_min = { DBL_MAX, DBL_MAX, DBL_MAX };
	vec3f local_max = { -DBL_MAX, -DBL_MAX, -DBL_MAX };

	// 2. 网格跨步循环 (Grid-Stride Loop)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < num_points; i += stride) {
		vec3f p = points[i];

		// 使用 fmin / fmax (针对 double)
		local_min.x = fmin(local_min.x, p.x);
		local_min.y = fmin(local_min.y, p.y);
		local_min.z = fmin(local_min.z, p.z);

		local_max.x = fmax(local_max.x, p.x);
		local_max.y = fmax(local_max.y, p.y);
		local_max.z = fmax(local_max.z, p.z);
	}

	// 3. Block 内共享内存归约
	// 注意：double 占用空间是 float 的两倍，Shared Memory 占用会增加
	// 256 线程 * 3 double * 8 bytes * 2 (min/max) = ~12 KB，完全没问题
	__shared__ vec3f s_min[256];
	__shared__ vec3f s_max[256];

	int tid = threadIdx.x;
	s_min[tid] = local_min;
	s_max[tid] = local_max;
	__syncthreads();

	// 树状归约
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			s_min[tid].x = fmin(s_min[tid].x, s_min[tid + s].x);
			s_min[tid].y = fmin(s_min[tid].y, s_min[tid + s].y);
			s_min[tid].z = fmin(s_min[tid].z, s_min[tid + s].z);

			s_max[tid].x = fmax(s_max[tid].x, s_max[tid + s].x);
			s_max[tid].y = fmax(s_max[tid].y, s_max[tid + s].y);
			s_max[tid].z = fmax(s_max[tid].z, s_max[tid + s].z);
		}
		__syncthreads();
	}

	// 4. 原子更新到全局内存
	// 使用我们自定义的 CAS 辅助函数
	if (tid == 0) {
		atomicMinDouble(&global_min->x, s_min[0].x);
		atomicMinDouble(&global_min->y, s_min[0].y);
		atomicMinDouble(&global_min->z, s_min[0].z);

		atomicMaxDouble(&global_max->x, s_max[0].x);
		atomicMaxDouble(&global_max->y, s_max[0].y);
		atomicMaxDouble(&global_max->z, s_max[0].z);
	}
}

void compute_global_aabb_double_cuda(vec3f* points, int num_points, vec3f* fmin, vec3f* fmax) {
	// 1. 初始化
	vec3f init_min = { DBL_MAX, DBL_MAX, DBL_MAX };
	vec3f init_max = { -DBL_MAX, -DBL_MAX, -DBL_MAX };

	cudaMemcpy(fmin, &init_min, sizeof(vec3f), cudaMemcpyHostToDevice);
	cudaMemcpy(fmax, &init_max, sizeof(vec3f), cudaMemcpyHostToDevice);

	// 2. 配置参数
	// ---------------------------------------------------------
	// 确定 Block Size
	// ---------------------------------------------------------
	const int blockSize = 256;

	// ---------------------------------------------------------
	// 确定 Grid Size (基于 SM 数量)
	// ---------------------------------------------------------
	int numSMs;
	int devId = 0;
	cudaGetDevice(&devId);
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);

	// 核心公式：每个 SM 分配 4 个 Block 以实现最大的占用率(Occupancy)
	// 对于 3070 Ti (48 SMs) -> grid = 192
	// 对于 5090 (假设 160 SMs) -> grid = 640
	// 这样能保证在两张卡上都能跑满，且利用 Grid-Stride Loop 自动平衡负载
	int optimalGridSize = numSMs * 4;

	// ---------------------------------------------------------
	// 3. 处理“数据量极小”的边界情况
	// ---------------------------------------------------------
	// 如果点云只有 100 个点，启动 640 个 Block 是浪费的。
	// 我们需要的 Block 数上限是 (N + blockSize - 1) / blockSize
	int neededBlocks = (num_points + blockSize - 1) / blockSize;

	// 取二者较小值
	int gridSize = std::min(optimalGridSize, neededBlocks);

	// 3. 启动
	compute_aabb_kernel_double << <gridSize, blockSize >> > (points, num_points, fmin, fmax);

	cudaDeviceSynchronize();
}

void compute_global_aabb_double_cpu(vec3f* points, int num_points, vec3f* fmin, vec3f* fmax) {
	vec3f init_min = { DBL_MAX, DBL_MAX, DBL_MAX };
	vec3f init_max = { -DBL_MAX, -DBL_MAX, -DBL_MAX };

	for (size_t i = 0; i < num_points; i++) {
		if (points[i].x < init_min.x)
			init_min.x = points[i].x;
		if (points[i].y < init_min.y)
			init_min.y = points[i].y;
		if (points[i].z < init_min.z)
			init_min.z = points[i].z;

		if (points[i].x > init_max.x)
			init_max.x = points[i].x;
		if (points[i].y > init_max.y)
			init_max.y = points[i].y;
		if (points[i].z > init_max.z)
			init_max.z = points[i].z;
	}
	fmin->x = init_min.x;
	fmin->y = init_min.y;
	fmin->z = init_min.z;

	fmax->x = init_max.x;
	fmax->y = init_max.y;
	fmax->z = init_max.z;
}


// 21 bits per axis, total 63 bits used in a 64-bit integer
// 这种方法通常被称为 "Magic Bits" 或者是使用表查找，这里使用位移法
__device__ __host__ __forceinline__ unsigned long long expandBits(unsigned int v) {
	// 1. 确保输入是 64 位宽，并只保留低 21 位
	unsigned long long ret = (unsigned long long)v & 0x1fffff;

	// 2. 开始位扩张 (Magic Bit Shifts)
	// 目标：将 21 位分散到 63 位中，每位之间隔两个 0

	// Step 1: 移动高 5 位到非常远的地方 (Bit 16-20 -> Bit 48-60)
	// 掩码解析: 0x1f00000000ffff
	// 低 16 位 (ffff) 保持不变
	// 中间 32 位 (00000000) 清零
	// 高 5 位 (1f) 对应移位后的位置
	ret = (ret | (ret << 32)) & 0x1f00000000ffffull;

	// Step 2: 处理中间层级 (Bit 8-15 移开)
	// 掩码解析: 0x1f0000ff0000ff
	// 使得每 16 位中有 8 位数据，8 位空隙
	ret = (ret | (ret << 16)) & 0x1f0000ff0000ffull;

	// Step 3: 继续细分 (Bit 4-7 移开)
	// 使得每 8 位中有 4 位数据，4 位空隙
	ret = (ret | (ret << 8)) & 0x100f00f00f00f00full;

	// Step 4: 使得每 4 位中有 2 位数据，2 位空隙
	ret = (ret | (ret << 4)) & 0x10c30c30c30c30c3ull;

	// Step 5: 最终一步，使得每 2 位中有 1 位数据，2 位空隙 (完成！)
	// 掩码 0x1249... 二进制末尾是 ...001001001
	ret = (ret | (ret << 2)) & 0x1249249249249249ull;

	return ret;
}

// 计算一个点的 3D Morton Code (Z-order curve) 的函数
__device__ __host__ __forceinline__ unsigned long long morton3D(double x, double y, double z,
	const vec3f min_box, const vec3f extent) {

	// 1. 归一化到 [0, 1]
	double nx = (x - min_box.x) / extent.x;
	double ny = (y - min_box.y) / extent.y;
	double nz = (z - min_box.z) / extent.z;

	// 2. 量化到 [0, 2^21 - 1] (即 0 到 2097151)
	// 使用 fmin 和 fmax 钳制范围，防止精度误差导致越界
	nx = fmin(fmax(nx * 2097152.0, 0.0), 2097151.0);
	ny = fmin(fmax(ny * 2097152.0, 0.0), 2097151.0);
	nz = fmin(fmax(nz * 2097152.0, 0.0), 2097151.0);

	unsigned int ux = (unsigned int)nx;
	unsigned int uy = (unsigned int)ny;
	unsigned int uz = (unsigned int)nz;

	// 3. 位交织
	// X 左移 0 位，Y 左移 1 位，Z 左移 2 位
	return expandBits(ux) | (expandBits(uy) << 1) | (expandBits(uz) << 2);
}

// 计算点集的 Morton Code 的 CUDA 核函数
__global__ void compute_morton_codes_kernel(const vec3f* points, int num_points,
	vec3f* min_box, vec3f* max_box,
	unsigned long long* morton_codes,
	int* object_ids) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (idx >= num_points) return;

	// 计算包围盒的边长
	vec3f extent;
	extent.x = max_box->x - min_box->x;
	extent.y = max_box->y - min_box->y;
	extent.z = max_box->z - min_box->z;

	// 防止除以 0
	if (extent.x <= 0) extent.x = 1.0;
	if (extent.y <= 0) extent.y = 1.0;
	if (extent.z <= 0) extent.z = 1.0;

	for (int i = idx; i < num_points; i += stride) {
		vec3f p = points[i];
		// 计算 Code
		morton_codes[i] = morton3D(p.x, p.y, p.z, *min_box, extent);
		// 顺便初始化索引数组，后续排序要跟着一起排
		object_ids[i] = i;
	}
}

// 计算点集的 Morton Code 的主机函数
void compute_morton_codes_cuda(const vec3f* points, int num_points,
	vec3f* min_box, vec3f* max_box,
	unsigned long long* morton_codes,
	int* object_ids) {

	// 配置参数
	const int blockSize = 256;
	int numSMs;
	int devId = 0;
	cudaGetDevice(&devId);
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
	int optimalGridSize = numSMs * 4;
	int neededBlocks = (num_points + blockSize - 1) / blockSize;
	// 取二者较小值
	int gridSize = std::min(optimalGridSize, neededBlocks);

	compute_morton_codes_kernel<<<gridSize, blockSize>>>(
		points, num_points,
		min_box, max_box,
		morton_codes,
		object_ids
	);

	cudaDeviceSynchronize();
}


// 辅助函数：计算下一个 2 的幂
int nextPowerOf2(int n) {
	if (n == 0) return 1;
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	return n + 1;
}

// CUDA 核函数：填充点集的 obj idx padding 部分
__global__ void set_padding_obj_index(int* object_ids, int num_points, int padded_num_points) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (idx >= padded_num_points) return;
	for (int i = idx; i < num_points; i += stride) {
		if (i >= num_points && i < padded_num_points)
			object_ids[i] = INT_MAX;
	}
}

// 填充点集的 obj idx padding 的主机函数
void set_padding_obj_index_cuda(int* object_ids, int num_points, int padded_num_points) {

	// 配置参数
	const int blockSize = 256;
	int numSMs;
	int devId = 0;
	cudaGetDevice(&devId);
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
	int optimalGridSize = numSMs * 4;
	int neededBlocks = (num_points + blockSize - 1) / blockSize;
	// 取二者较小值
	int gridSize = std::min(optimalGridSize, neededBlocks);

	set_padding_obj_index<<<gridSize, blockSize>>>(
		object_ids, num_points, padded_num_points
	);

	cudaDeviceSynchronize();
}

// 交换两个元素 (Key 和 Value 一起交换)
__device__ void swap_kv(unsigned long long* keys, int* values, int a, int b) {
	unsigned long long tmpKey = keys[a];
	keys[a] = keys[b];
	keys[b] = tmpKey;

	int tmpVal = values[a];
	values[a] = values[b];
	values[b] = tmpVal;
}

__global__ void bitonic_sort_step_arbitrary_n(unsigned long long* keys, int* values, int num_points, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    // 1. 只有当 ixj > i 时才由当前线程处理（每对只处理一次）
    if (ixj > i) {
        
        // 2. 关键修改：边界检查
        // 如果 i 越界，直接退出（不该发生，只要 gridSize 算对）
        if (i >= num_points) return;
        
        // 如果 ixj 越界，我们假想 keys[ixj] 是最大值。
        // 在升序排序中，任何值都小于最大值，所以无需交换，直接不需要处理。
        if (ixj >= num_points) return;

        // 3. 正常的比较交换逻辑
        bool ascending = (i & k) == 0;
        unsigned long long key_i = keys[i];
        unsigned long long key_ixj = keys[ixj];

        if (ascending) {
            if (key_i > key_ixj) {
                swap_kv(keys, values, i, ixj);
            }
        } else {
            if (key_i < key_ixj) {
                swap_kv(keys, values, i, ixj);
            }
        }
    }
}

// Bitonic Sort 入口函数
void bitonic_sort_cuda(unsigned long long* d_keys, int* d_values, int num_points) {
    // 虽然不用填充数据，但排序的阶段（Stage）必须按照 2 的幂次走
    int n_power2 = nextPowerOf2(num_points);

    int threadsPerBlock = 256;
    // 启动线程数仍然需要覆盖 n_power2 / 2 对比较
    // 实际上只要覆盖 num_points 即可，因为越界的全都被上面 kernel 中的 check 过滤了
    // 但为了逻辑保险，保持足够多的线程
    int blocksPerGrid = (n_power2 / 2 + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0) blocksPerGrid = 1;

    for (int k = 2; k <= n_power2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step_arbitrary_n<<<blocksPerGrid, threadsPerBlock>>>(
                d_keys, d_values, num_points, j, k
            );
        }
    }
    cudaDeviceSynchronize();
}

REAL checkDistCuda(const kmesh* m1, const kmesh* m2, std::vector<id_pair>& rets, CheckMode mode) {
	// 报错信息
	cudaError_t err = cudaSuccess;

	// 准备计时
	float all_start_time_cpu = omp_get_wtime();

	// CUDA 计时器

	// 获取网格的顶点数量
	unsigned int pointNumber1 = m1->_num_vtx;
	unsigned int pointNumber2 = m2->_num_vtx;

	// 获取数据大小
	size_t size1 = pointNumber1 * sizeof(vec3f);
	size_t size2 = pointNumber2 * sizeof(vec3f);

	size_t allPointsSize = pointNumber1 * pointNumber2;
	size_t distSize = allPointsSize * sizeof(double);

	size_t pairDataSize = 2 * sizeof(vec3f);

	// CUDA设备内存指针
	// 原始数据：两个点集
	vec3f* points1Cuda = nullptr;
	vec3f* points2Cuda = nullptr;

	// 计算结果：距离矩阵
	double* distDataCuda = nullptr;

	// AABB包围盒
	vec3f* aabbCuda1 = nullptr;
	vec3f* aabbCuda2 = nullptr;

	// morton code
	unsigned long long* mortonCodes1Cuda = nullptr;
	unsigned long long* mortonCodes2Cuda = nullptr;

	// 用于mortoncode的排序索引
	int * ObjectIDx1Cuda = nullptr;
	int* ObjectIDx2Cuda = nullptr;

	// 转移到主存里的 结果数据
	double* distDataHost = nullptr;

	// 对内存分配计时开始
	float start_Malloc_cpu = omp_get_wtime();

	// 开辟CUDA设备内存
	err = cudaMalloc((void**)&points1Cuda, size1);
	CUDA_CHECK_ERROR(err);

	err = cudaMalloc((void**)&points2Cuda, size2);
	CUDA_CHECK_ERROR(err);

	if (mode == CHECK_MODE_BRUTE_FORCE)
	{
		// 暴力模式下的内存分配
		err = cudaMalloc((void**)&distDataCuda, distSize);
		CUDA_CHECK_ERROR(err);

		distDataHost = (double*)malloc(distSize);
	}
	else if (mode == CHECK_MODE_BVH) {
		// BVH模式下的内存分配
		// AABB内存分配
		err = cudaMalloc((void**)&aabbCuda1, 2 * sizeof(vec3f));
		CUDA_CHECK_ERROR(err);
		err = cudaMalloc((void**)&aabbCuda2, 2 * sizeof(vec3f));
		CUDA_CHECK_ERROR(err);
		// 莫顿码内存分配
		err = cudaMalloc((void**)&mortonCodes1Cuda, pointNumber1 * sizeof(unsigned long long));
		CUDA_CHECK_ERROR(err);
		err = cudaMalloc((void**)&mortonCodes2Cuda, pointNumber2 * sizeof(unsigned long long));
		CUDA_CHECK_ERROR(err);
		// 索引内存分配
		// 注意：这里分配的大小是下一个2的幂，以便后续排序算法使用
		err = cudaMalloc((void**)&ObjectIDx1Cuda, pointNumber1 * sizeof(int));
		CUDA_CHECK_ERROR(err);
		err = cudaMalloc((void**)&ObjectIDx2Cuda, pointNumber2 * sizeof(int));
		CUDA_CHECK_ERROR(err);
	}

	// 对数据传输计时开始
	double start_DataTransform_cpu = omp_get_wtime();
	
	// 复制数据到设备
	cudaMemcpy(points1Cuda, m1->_vtxs, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(points2Cuda, m2->_vtxs, size2, cudaMemcpyHostToDevice);

	// 对计算核函数计时开始
	double start_Kernel_cpu = omp_get_wtime();

	// 最后结果
	unsigned idx1 = 0, idx2 = 0;
	double minDist = FLT_MAX;
	if (mode == CHECK_MODE_BRUTE_FORCE) {
		// 启动：暴力计算所有点对距离
		dim3 block(16, 16, 1);
		dim3 thread((pointNumber1 + block.x - 1) / block.x, (pointNumber2 + block.y - 1) / block.y, 1);
		BruteForce_CalculateDistance << < thread, block >> > (points1Cuda, points2Cuda, distDataCuda, pointNumber1, pointNumber2);

		// 将数据从设备复制回主机
		cudaMemcpy(distDataHost, distDataCuda, distSize, cudaMemcpyDeviceToHost);

		// 处理结果数据
		// 初始解法：直接计算所有点对距离，找出最小距离
		for (unsigned int i = 0; i < allPointsSize; i++) {
			if (distDataHost[i] < minDist) {
				idx1 = i % pointNumber1;
				idx2 = i / pointNumber1;
				minDist = distDataHost[i];
			}
		}
		rets.clear();
		rets.push_back(id_pair(idx1, idx2, false));
	}
	else if (mode == CHECK_MODE_BVH) {
		// BVH模式下的距离计算
		compute_global_aabb_double_cuda(points1Cuda, pointNumber1, aabbCuda1, aabbCuda1 + 1);
		compute_global_aabb_double_cuda(points2Cuda, pointNumber2, aabbCuda2, aabbCuda2 + 1);

		// debug: 将AABB从设备复制回主机
		vec3f aabbCpu1[2];
		vec3f aabbCpu2[2];
		cudaMemcpy(aabbCpu1, aabbCuda1, 2 * sizeof(vec3f), cudaMemcpyDeviceToHost);
		cudaMemcpy(aabbCpu2, aabbCuda2, 2 * sizeof(vec3f), cudaMemcpyDeviceToHost);

		// test expand函数
		unsigned int testUnexpanded = 2097151;
		auto testExpanded = expandBits(testUnexpanded);

		// 计算morton code
		compute_morton_codes_cuda(points1Cuda, pointNumber1,
			&(aabbCuda1[0]), &(aabbCuda1[1]),
			mortonCodes1Cuda,
			ObjectIDx1Cuda);
		compute_morton_codes_cuda(points2Cuda, pointNumber2,
			&(aabbCuda2[0]), &(aabbCuda2[1]),
			mortonCodes2Cuda,
			ObjectIDx2Cuda);

		// 排序morton code和对应的index
		/*bitonic_sort_cuda(mortonCodes1Cuda, ObjectIDx1Cuda, nextPowerOf2(pointNumber1));
		bitonic_sort_cuda(mortonCodes2Cuda, ObjectIDx2Cuda, nextPowerOf2(pointNumber2));*/

		// debug：将morton code从设备复制回主机（全部）
		vector<unsigned long long> mc1cuda;
		mc1cuda.resize(pointNumber1);
		cudaMemcpy(mc1cuda.data(), mortonCodes1Cuda, pointNumber1, cudaMemcpyDeviceToHost);
		unsigned long long mc1cpu = morton3D(m1->_vtxs[0].x, m1->_vtxs[0].y, m1->_vtxs[0].z,
			aabbCpu1[0], aabbCpu1[1] - aabbCpu1[0]);

		rets.clear();
		rets.push_back(id_pair(0, 0, false));
	}

	// 对计算核函数计时结束
	double stop_Kernel_cpu = omp_get_wtime();

	// 释放CUDA设备内存
	cudaFree(points1Cuda);
	cudaFree(points2Cuda);
	if (mode == CHECK_MODE_BRUTE_FORCE) {
		cudaFree(distDataCuda);

		// 释放主机内存
		free(distDataHost);
	}
	else if (mode == CHECK_MODE_BVH) {
		cudaFree(aabbCuda1);
		cudaFree(aabbCuda2);  
		cudaFree(mortonCodes1Cuda);
		cudaFree(mortonCodes2Cuda);
		cudaFree(ObjectIDx1Cuda);
		cudaFree(ObjectIDx2Cuda);
	}


	float all_stop_time_cpu = omp_get_wtime();

	std::cout << "\nPrepare Time: " << (start_Malloc_cpu - all_start_time_cpu) * 1000.0 << " ms\n";
	std::cout << "CPU Malloc Time: " << (start_DataTransform_cpu - start_Malloc_cpu) * 1000.0 << " ms\n";
	std::cout << "CPU Data Transform Time: " << (start_Kernel_cpu - start_DataTransform_cpu) * 1000.0 << " ms\n";
	std::cout << "CPU Kernel Time: " << (stop_Kernel_cpu - start_Kernel_cpu) * 1000.0 << " ms\n";
	std::cout << "CPU Free Time: " << (all_stop_time_cpu - stop_Kernel_cpu) * 1000.0 << " ms\n";
	std::cout << "\nTotal CUDA Time: " << (all_stop_time_cpu - all_start_time_cpu) * 1000.0 << " ms\n\n";

	return minDist;
}