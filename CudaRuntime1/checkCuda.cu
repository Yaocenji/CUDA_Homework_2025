

#include "checkCuda.h"

// Thrust 排序函数声明
extern "C" void perform_thrust_sort(unsigned long long* d_keys, int* d_values, int num_points);

// Thrust 查询最小值函数声明
extern "C" void perform_thrust_min_element(double* d_data, int num_points, double* out_min_val, int* out_idx);

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



// CUDA 核函数：填充点集的 padding 部分，包括morton code和obj idx
__global__ void set_padding_data(unsigned long long* morton_codes, int* object_ids,
	int num_points, int padded_num) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + num_points;
	if (idx < padded_num) {
		morton_codes[idx] = 0xFFFFFFFFFFFFFFFFULL; // ULLONG_MAX
		object_ids[idx] = -1; // 或者 INT_MAX，表示无效ID
	}
}

// 填充点集的 obj idx padding 的主机函数
void set_padding_cuda(unsigned long long* morton_codes, int* object_ids, int num_points, int padded_num_points) {

	// 配置参数
	const int blockSize = 256;
	int gridSize = (padded_num_points + blockSize - 1) / blockSize;

	set_padding_data <<<gridSize, blockSize>>>(
		morton_codes, object_ids, num_points, padded_num_points
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

__global__ void bitonic_sort_step(unsigned long long* keys, int* values, int j, int k) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int ixj = i ^ j;

	// 标准 Bitonic Sort 逻辑：只要 i < ixj 就处理
	// 因为显存已经分配到了 nextPowerOf2，所以不会越界
	if (ixj > i) {
		bool ascending = (i & k) == 0;
		unsigned long long key_i = keys[i];
		unsigned long long key_ixj = keys[ixj];

		// 升序逻辑
		if (ascending) {
			if (key_i > key_ixj) {
				swap_kv(keys, values, i, ixj);
			}
		}
		// 降序逻辑 (Padding 的 ULLONG_MAX 会在这里被交换到前面，这是算法必须的)
		else {
			if (key_i < key_ixj) {
				swap_kv(keys, values, i, ixj);
			}
		}
	}
}

// Bitonic Sort 入口函数
void bitonic_sort_cuda(unsigned long long* d_keys, int* d_values, int num_points) {
	// 执行标准双调排序
	int padded_n = nextPowerOf2(num_points);
	int threadsPerBlock = 256;
	int blocksPerGrid = (padded_n / 2 + threadsPerBlock - 1) / threadsPerBlock;
	if (blocksPerGrid == 0) blocksPerGrid = 1; // 防止 n=1 时的错误

	for (int k = 2; k <= padded_n; k <<= 1) {
		for (int j = k >> 1; j > 0; j >>= 1) {
			bitonic_sort_step << <blocksPerGrid, threadsPerBlock >> > (
				d_keys, d_values, j, k
				);
		}
	}
	cudaDeviceSynchronize();
}

// 计算两个位置 Morton Code 的最长公共前缀长度 (LCP)
__device__ __forceinline__ int delta(
	const unsigned long long* sorted_morton_codes,
	int num_points,
	int i,
	int j) {
	// 边界处理
	if (j < 0 || j >= num_points) return -1;

	unsigned long long code_i = sorted_morton_codes[i];
	unsigned long long code_j = sorted_morton_codes[j];

	// 如果 Code 不同，计算 XOR 后的前导零个数
	if (code_i != code_j) {
		return __clzll(code_i ^ code_j);
	}
	else {
		// 如果 Code 相同，使用索引作为 Tie-breaker
		// 加上 64 是为了让它比任何 Code 不相同的情况都有更大的 LCP
		// 这样相同的 Code 会被聚在一起处理
		return 64 + __clzll((unsigned long long)i ^ (unsigned long long)j);
	}
}

__global__ void initialize_bvh_nodes_cuda(LBVHNode * bvh_nodes, int bvh_number) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (idx >= bvh_number) return;

	for (int i = idx; i < bvh_number; i += stride) {
		bvh_nodes[i].left_child = -1;
		bvh_nodes[i].right_child = -1;
		bvh_nodes[i].parent = -1;
		bvh_nodes[i].atom_flag = 0;
	}
}

void initialize_bvh_nodes(LBVHNode* bvh_nodes, int bvh_number) {
	// 配置参数
	const int blockSize = 256;
	int numSMs;
	int devId = 0;
	cudaGetDevice(&devId);
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
	int optimalGridSize = numSMs * 4;
	int neededBlocks = (bvh_number + blockSize - 1) / blockSize;
	// 取二者较小值
	int gridSize = std::min(optimalGridSize, neededBlocks);

	initialize_bvh_nodes_cuda<<<gridSize, blockSize>>>(
		bvh_nodes, bvh_number
	);

	cudaDeviceSynchronize();
}

/// <summary>
/// 构建 BVH 层次结构的 CUDA 核函数
/// </summary>
/// <param name="sorted_morton_codes"></param>
/// <param name="sorted_object_ids"></param>
/// <param name="leaf_nodes"></param>
/// <param name="internal_nodes"></param>
/// <param name="num_points"></param>
/// <returns></returns>
__global__ void generate_hierarchy_kernel(
	const unsigned long long* sorted_morton_codes,
	const int* sorted_object_ids, // 这里虽然暂时没用到，但后续 Refit 可能需要知道叶子对应的原始对象
	LBVHNode* leaf_nodes,         // 长度 N
	LBVHNode* internal_nodes,     // 长度 N - 1
	int num_points) {
	// 每个线程处理一个内部节点 (0 到 N-2)
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_points - 1) return;


	// -------------------------------------------------------------------------
	// 1. 确定范围的方向 (Direction)
	// 比较 i 与 i+1 的 LCP，以及 i 与 i-1 的 LCP，看往哪边延伸更长
	// -------------------------------------------------------------------------
	int d_diff = delta(sorted_morton_codes, num_points, i, i + 1) -
		delta(sorted_morton_codes, num_points, i, i - 1);

	// d = 1 表示向右 (+1), d = -1 表示向左 (-1)
	int d = (d_diff > 0) ? 1 : -1;

	// 计算当前节点 i 的最小 LCP (min_delta)
	int min_delta = delta(sorted_morton_codes, num_points, i, i - d);

	// -------------------------------------------------------------------------
	// 2. 确定范围的另一端 (l_max)
	// 我们需要找到一个位置，使得 delta(i, l_max) > min_delta
	// 使用“指数步长”搜索
	// -------------------------------------------------------------------------
	int l_max = 2;
	while (delta(sorted_morton_codes, num_points, i, i + l_max * d) > min_delta) {
		l_max *= 2;
		if (l_max > num_points)
			break;
	}

	// 二分查找精确位置
	int l = 0;
	for (int t = l_max / 2; t >= 1; t /= 2) {
		if (delta(sorted_morton_codes, num_points, i, i + (l + t) * d) > min_delta) {
			l += t;
		}
	}

	int j = i + l * d; // j 就是范围的另一端

	// 确保 delta_node 是 i 和 j 之间的最大公共前缀长度
	int delta_node = delta(sorted_morton_codes, num_points, i, j);

	// -------------------------------------------------------------------------
	// 3. 寻找分割点 (Split Position / Gamma)
	// 在范围 [min(i,j), max(i,j)] 内找到分割点 gamma
	// 使得 range 被分为 [first, gamma] 和 [gamma+1, last]
	// -------------------------------------------------------------------------
	int s = 0;
	// 步长搜索
	// 范围是被压缩在 [i, i + l*d] 之间的 (或者反过来)
	// 我们需要计算 binary search 的步长
	// 这里的逻辑稍微复杂，因为 l 可能很大

	// 重新定义范围的左右边界 [first, last]
	int first = min(i, j);
	int last = max(i, j);

	if (first == last) {
		// 理论上不会发生，除非 N=1
		return;
	}

	// 在 [first, last] 之间寻找 gamma
	// gamma 是使得 delta(first, gamma) == delta_node 的最大索引
	// 我们使用 common prefix >= delta_node 来判断

	// 使用 Karras 的优化二分查找
	int split = first;
	int step = last - first;

	do {
		step = (step + 1) >> 1; // ceil(step / 2)
		int new_split = split + step;
		if (new_split < last) {
			if (delta(sorted_morton_codes, num_points, first, new_split) > delta_node) {
				split = new_split;
			}
		}
	} while (step > 1);

	// split 就是 gamma

	// -------------------------------------------------------------------------
	// 4. 构建拓扑连接
	// -------------------------------------------------------------------------
	// 当前内部节点的索引 (在全局视角下，通常内部节点索引偏移 N)
	// 但这里我们使用独立的 internal_nodes 数组，索引就是 i
	// 为了方便统一管理，我们在存储 child 索引时使用全局索引约定：
	// [0, N-1] -> Leaf
	// [N, 2N-2] -> Internal (对应 internal_nodes 数组的下标 0 到 N-2)

	int current_node_idx = num_points + i; // 全局索引

	int left_idx = split;
	int right_idx = split + 1;

	// 左孩子
	if (min(i, j) == left_idx) {
		// 左孩子是叶子
		// 这里的 leaf_nodes 对应 sorted 后的第 left_idx 个元素
		leaf_nodes[left_idx].parent = current_node_idx;
		internal_nodes[i].left_child = left_idx;
	}
	else {
		// 左孩子是内部节点
		// 内部节点 idx = N + left_idx
		internal_nodes[left_idx].parent = current_node_idx; // 注意这里是对 left_idx 的节点写 parent
		internal_nodes[i].left_child = num_points + left_idx;
	}

	// 右孩子
	if (max(i, j) == right_idx) {
		// 右孩子是叶子
		leaf_nodes[right_idx].parent = current_node_idx;
		internal_nodes[i].right_child = right_idx;
	}
	else {
		// 右孩子是内部节点
		internal_nodes[right_idx].parent = current_node_idx;
		internal_nodes[i].right_child = num_points + right_idx;
	}
}

/// <summary>
/// 构建BVH结构的主机函数
/// </summary>
/// <param name="d_sorted_morton_codes"></param>
/// <param name="d_sorted_object_ids"></param>
/// <param name="d_leaf_nodes"></param>
/// <param name="d_internal_nodes"></param>
/// <param name="num_points"></param>
void build_lbvh_structure_cuda(
	unsigned long long* d_sorted_morton_codes,
	int* d_sorted_object_ids,
	LBVHNode* d_leaf_nodes,
	LBVHNode* d_internal_nodes,
	int num_points) {
	// 配置 Kernel
	int threadsPerBlock = 256;
	// 注意：我们只需要 N-1 个线程来创建内部节点
	int blocksPerGrid = (num_points - 1 + threadsPerBlock - 1) / threadsPerBlock;

	generate_hierarchy_kernel << <blocksPerGrid, threadsPerBlock >> > (
		d_sorted_morton_codes,
		d_sorted_object_ids,
		d_leaf_nodes,
		d_internal_nodes,
		num_points
		);

	cudaDeviceSynchronize();
}


// 查找根节点
__global__ void find_root_cuda(LBVHNode* internal_nodes, int num_points, int* theRoot, int* rootCount) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if (idx >= num_points - 1) return;

	for (int i = idx; i < num_points - 1; i += stride) {
		if (internal_nodes[i].parent == -1) {
			atomicAdd(rootCount, 1);
			*theRoot = i + num_points;
		}
	}
}

void find_root(LBVHNode* internal_nodes, int num_points, int* theRoot, int* rootCount) {
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

	find_root_cuda << <gridSize, blockSize >> > (
		internal_nodes, num_points, theRoot, rootCount
		);

	cudaDeviceSynchronize();
}


// 辅助函数：中序遍历验证
bool verify_lbvh_structure(const std::vector<LBVHNode>& nodes, int num_points) {
	// 1. 寻找根节点
	// 根节点是唯一一个 parent 为 -1 的内部节点
	// 注意：内部节点的索引范围在 nodes 数组中是 [num_points, 2*num_points - 2]
	int root_idx = -1;
	for (int i = num_points; i < 2 * num_points - 1; ++i) {
		if (nodes[i].parent == -1) {
			if (root_idx != -1) {
				std::cerr << "Error: Found multiple roots! (" << root_idx << ", " << i << ")\n";
				return false;
			}
			root_idx = i;
		}
	}

	if (root_idx == -1) {
		std::cerr << "Error: No root found! Cyclic dependency or initialization error.\n";
		return false;
	}

	std::cout << "Root found at index: " << root_idx << "\n";

	// 2. 中序遍历收集叶子节点
	std::vector<int> leaves;
	std::stack<int> s;
	int curr = root_idx;

	// 迭代式中序遍历
	while (curr != -1 || !s.empty()) {
		// 向左走到尽头
		while (curr != -1) {
			// 如果是叶子节点 (索引 < num_points)
			if (curr < num_points) {
				leaves.push_back(curr);
				curr = -1; // 叶子没有孩子，相当于到了尽头
			}
			else {
				// 内部节点
				s.push(curr);
				curr = nodes[curr].left_child;
			}
		}

		if (!s.empty()) {
			curr = s.top();
			s.pop();
			// 转向右子树
			curr = nodes[curr].right_child;
		}
	}

	// 3. 验证序列是否为 0, 1, 2 ... N-1
	if (leaves.size() != num_points) {
		std::cerr << "Error: Leaf count mismatch! Expected " << num_points << ", got " << leaves.size() << "\n";
		return false;
	}

	for (int i = 0; i < num_points; ++i) {
		if (leaves[i] != i) {
			std::cerr << "Error: Sequence mismatch at index " << i << ". Expected " << i << ", got " << leaves[i] << "\n";
			// 输出上下文帮助 debug
			int start = std::max(0, i - 5);
			int end = std::min(num_points, i + 5);
			std::cerr << "Context: ";
			for (int k = start; k < end; ++k) std::cerr << leaves[k] << " ";
			std::cerr << "\n";
			return false;
		}
	}

	std::cout << "Verification Passed! LBVH topology is correct.\n";
	return true;
}



// 辅助函数：合并两个 AABB
__device__ __forceinline__ void aabb_union(const vec3f& min_a, const vec3f& max_a,
	const vec3f& min_b, const vec3f& max_b,
	vec3f& res_min, vec3f& res_max) {
	res_min.x = fmin(min_a.x, min_b.x);
	res_min.y = fmin(min_a.y, min_b.y);
	res_min.z = fmin(min_a.z, min_b.z);

	res_max.x = fmax(max_a.x, max_b.x);
	res_max.y = fmax(max_a.y, max_b.y);
	res_max.z = fmax(max_a.z, max_b.z);
}

__global__ void refit_lbvh_kernel(
	const int* sorted_object_ids,
	const vec3f* points,
	LBVHNode* leaf_nodes,      // 长度 N
	LBVHNode* internal_nodes,  // 长度 N-1
	int num_points) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_points) return;

	// -------------------------------------------------------------------------
	// 1. 初始化叶子节点 AABB
	// -------------------------------------------------------------------------
	int obj_id = sorted_object_ids[idx];
	// 注意：如果使用了 Padding，后面的 obj_id 是 -1 或无效值
	// 但我们的 idx < num_points 保证了只处理有效点

	vec3f p = points[obj_id];

	// 叶子节点的包围盒就是点本身 (min=p, max=p)
	// 为了防止数值误差，也可以稍微扩一点点，但在点云距离计算中通常不需要
	leaf_nodes[idx].min_box = p;
	leaf_nodes[idx].max_box = p;

	// -------------------------------------------------------------------------
	// 2. 自底向上更新
	// -------------------------------------------------------------------------
	int current_node_idx = idx; // 当前处理的节点索引（一开始是叶子）
	int parent_idx = leaf_nodes[idx].parent;

	while (parent_idx != -1) { // 直到根节点（根节点的 parent 是 -1）

		// internal_nodes 数组是从 0 开始的，但 parent_idx 是全局索引 (>= N)
		// 所以要偏移
		int internal_array_idx = parent_idx - num_points;

		// 使用 atomicAdd 来进行同步
		// atom_flag 在初始化时必须全为 0
		int old_val = atomicAdd(&internal_nodes[internal_array_idx].atom_flag, 1);

		// ---------------------------------------------------------------------
		// Case A: 我是第一个到达父节点的线程
		// ---------------------------------------------------------------------
		if (old_val == 0) {
			// 我的工作完成了，剩下的交给我的兄弟。
			// 此时不需要写入 AABB，因为兄弟线程会读取我的 AABB。
			// 必须确保我的 AABB 写入操作对兄弟可见（Global Memory Coherency）
			// 在 CUDA 中，同一个 Kernel 内的 Global Memory 写入在 atomic 操作确立顺序后通常是可见的，
			// 但加上 threadfence 是个好习惯
			__threadfence(); 
			return;
		}

		// ---------------------------------------------------------------------
		// Case B: 我是第二个到达父节点的线程 (old_val == 1)
		// ---------------------------------------------------------------------
		// 说明左右孩子都 ready 了。我负责计算父节点的 AABB。

		// 1. 获取左右孩子的索引
		int left_child = internal_nodes[internal_array_idx].left_child;
		int right_child = internal_nodes[internal_array_idx].right_child;

		// 2. 读取左右孩子的 AABB
		vec3f l_min, l_max, r_min, r_max;

		// 读取左孩子
		if (left_child < num_points) { // 是叶子
			l_min = leaf_nodes[left_child].min_box;
			l_max = leaf_nodes[left_child].max_box;
		}
		else { // 是内部节点
			l_min = internal_nodes[left_child - num_points].min_box;
			l_max = internal_nodes[left_child - num_points].max_box;
		}

		// 读取右孩子
		if (right_child < num_points) { // 是叶子
			r_min = leaf_nodes[right_child].min_box;
			r_max = leaf_nodes[right_child].max_box;
		}
		else { // 是内部节点
			r_min = internal_nodes[right_child - num_points].min_box;
			r_max = internal_nodes[right_child - num_points].max_box;
		}

		// 3. 计算并集并写入当前父节点
		aabb_union(l_min, l_max, r_min, r_max,
			internal_nodes[internal_array_idx].min_box,
			internal_nodes[internal_array_idx].max_box);

		// 4. 继续向上攀升
		current_node_idx = parent_idx;
		parent_idx = internal_nodes[internal_array_idx].parent;

		// 确保写入完成，再进入下一轮原子操作
		// __threadfence(); 
	}
}

void refit_lbvh_structure_cuda(
	int* d_sorted_object_ids,
	vec3f* d_points,
	LBVHNode* d_leaf_nodes,
	LBVHNode* d_internal_nodes,
	int num_points) {
	// 线程数等于有效点数
	int threadsPerBlock = 256;
	int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

	refit_lbvh_kernel << <blocksPerGrid, threadsPerBlock >> > (
		d_sorted_object_ids,
		d_points,
		d_leaf_nodes,
		d_internal_nodes,
		num_points
		);

	cudaDeviceSynchronize();
}


// 辅助函数：HOST端验证递归填充AABB正确性
// 辅助：检查 Box A 是否包含 Box B
bool is_contained(const vec3f& a_min, const vec3f& a_max,
	const vec3f& b_min, const vec3f& b_max) {
	// 使用一个小 epsilon 防止浮点误差，虽然后续遍历计算理应是严格的
	const float eps = 1e-5f;

	if (a_min.x > b_min.x + eps) return false;
	if (a_min.y > b_min.y + eps) return false;
	if (a_min.z > b_min.z + eps) return false;

	if (a_max.x < b_max.x - eps) return false;
	if (a_max.y < b_max.y - eps) return false;
	if (a_max.z < b_max.z - eps) return false;

	return true;
}

// 递归检查节点
bool check_node_aabb_recursive(const std::vector<LBVHNode>& nodes, int curr_idx, int num_points) {
	// 1.如果是叶子节点
	if (curr_idx < num_points) {
		// 叶子节点的逻辑已经在初始化时决定了（min=max=point），
		// 只要不是无穷大或乱码即可。
		// 这里简单检查 min <= max
		const auto& node = nodes[curr_idx];
		if (node.min_box.x > node.max_box.x ||
			node.min_box.y > node.max_box.y ||
			node.min_box.z > node.max_box.z) {
			std::cerr << "Error: Leaf Node " << curr_idx << " has invalid AABB (min > max)!\n";
			return false;
		}
		return true;
	}

	// 2. 如果是内部节点
	const auto& node = nodes[curr_idx];
	int left_idx = node.left_child;
	int right_idx = node.right_child;

	if (left_idx == -1 || right_idx == -1) {
		std::cerr << "Error: Internal Node " << curr_idx << " has invalid children indices!\n";
		return false;
	}

	// 获取孩子的引用
	// 注意：这里的 nodes 包含了叶子和内部节点，直接用下标访问
	const auto& l_node = nodes[left_idx];
	const auto& r_node = nodes[right_idx];

	// 检查：父节点必须包含左孩子
	if (!is_contained(node.min_box, node.max_box, l_node.min_box, l_node.max_box)) {
		std::cerr << "Error: Internal Node " << curr_idx << " does NOT contain Left Child " << left_idx << "!\n";
		std::cerr << "  Parent: [" << node.min_box.x << ", " << node.max_box.x << "]\n";
		std::cerr << "  Child : [" << l_node.min_box.x << ", " << l_node.max_box.x << "]\n";
		return false;
	}

	// 检查：父节点必须包含右孩子
	if (!is_contained(node.min_box, node.max_box, r_node.min_box, r_node.max_box)) {
		std::cerr << "Error: Internal Node " << curr_idx << " does NOT contain Right Child " << right_idx << "!\n";
		return false;
	}

	// 递归检查子树
	if (!check_node_aabb_recursive(nodes, left_idx, num_points)) return false;
	if (!check_node_aabb_recursive(nodes, right_idx, num_points)) return false;

	return true;
}

// 入口函数
bool verify_bvh_aabb(const std::vector<LBVHNode>& nodes, int num_points) {
	// 1. 寻找根节点 (parent == -1)
	int root_idx = -1;
	// 内部节点从 num_points 开始
	for (int i = num_points; i < 2 * num_points - 1; ++i) {
		if (nodes[i].parent == -1) {
			if (root_idx != -1) {
				std::cerr << "Error: Multiple roots found during AABB check.\n";
				return false;
			}
			root_idx = i;
		}
	}

	if (root_idx == -1) {
		std::cerr << "Error: No root found.\n";
		return false;
	}

	std::cout << "Starting AABB integrity check from Root " << root_idx << "...\n";

	// 2. 开始递归检查
	if (check_node_aabb_recursive(nodes, root_idx, num_points)) {
		std::cout << "AABB Verification Passed! Hierarchy is geometrically valid.\n";
		return true;
	}
	else {
		std::cout << "AABB Verification Failed.\n";
		return false;
	}
}



// 计算点到点的距离平方
__device__ __forceinline__ double dist_sq_point_point(const vec3f& a, const vec3f& b) {
	double dx = a.x - b.x;
	double dy = a.y - b.y;
	double dz = a.z - b.z;
	return dx * dx + dy * dy + dz * dz;
}

// 计算点 P 到 AABB (min_box, max_box) 的最小距离平方
// 如果点在盒子里，距离为 0
__device__ __forceinline__ double dist_sq_point_aabb(const vec3f& p, const vec3f& min_box, const vec3f& max_box) {
	double dx = 0.0, dy = 0.0, dz = 0.0;

	if (p.x < min_box.x) dx = min_box.x - p.x;
	else if (p.x > max_box.x) dx = p.x - max_box.x;

	if (p.y < min_box.y) dy = min_box.y - p.y;
	else if (p.y > max_box.y) dy = p.y - max_box.y;

	if (p.z < min_box.z) dz = min_box.z - p.z;
	else if (p.z > max_box.z) dz = p.z - max_box.z;

	return dx * dx + dy * dy + dz * dz;
}


// 最后的函数：查询距离
__global__ void query_distance_kernel(
	const vec3f* query_points,    // 点云 A (查询发起者)
	int num_query_points,
	const LBVHNode* bvh_nodes,    // 点云 B 的 BVH 数组
	const int* sorted_obj_ids_b,  // 点云 B 的排序索引 (用于从 points_b 获取坐标)
	const vec3f* points_b_raw,    // 点云 B 的原始坐标数据
	int* root_node_idx,           // 点云 B 的根节点索引
	int num_bvh_points,           // 点云 B 的点数 (用于判断 is_leaf)
	double* out_min_dists,        // 输出：每个查询点的最近距离
	int* out_min_indices)         // 输出：每个查询点的最近对应点的索引
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_query_points) return;

    vec3f q = query_points[idx];
    double current_min_sq = DBL_MAX;
    
    // [新增] 本地变量记录当前找到的最近点的原始 ID
    int current_best_idx = -1; 

    // 手动栈
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = *root_node_idx;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        LBVHNode node = bvh_nodes[node_idx];

        // 1. AABB 剔除
        double box_dist_sq = dist_sq_point_aabb(q, node.min_box, node.max_box);
        if (box_dist_sq >= current_min_sq) continue;

        // 2. 叶子节点
        if (node_idx < num_bvh_points) {
            // AABB 验证正确后，直接用 AABB 中心作为点坐标
            double leaf_dist_sq = dist_sq_point_point(q, node.min_box);

            if (leaf_dist_sq < current_min_sq) {
                current_min_sq = leaf_dist_sq;
                
                // [新增] 关键步骤：记录这个点的原始索引
                // node_idx 是排序后的叶子索引，通过 sorted_obj_ids_b 找回原始 ID
                current_best_idx = sorted_obj_ids_b[node_idx];
            }
        }
        // 3. 内部节点
        else {
            int left = node.left_child;
            int right = node.right_child;
            if (left != -1) stack[stack_ptr++] = left;
            if (right != -1) stack[stack_ptr++] = right;
        }
    }

    out_min_dists[idx] = sqrt(current_min_sq);
    out_min_indices[idx] = current_best_idx;
}



REAL checkDistCuda(const kmesh* m1, const kmesh* m2, std::vector<id_pair>& rets, CheckMode mode, bool stepTimeRecord) {
	// 报错信息
	cudaError_t err = cudaSuccess;

#ifndef PROF
	// 准备计时
	float all_start_time_cpu = omp_get_wtime();
#endif

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

	// LBVH节点
	LBVHNode* BVHNodes1Cuda = nullptr;
	LBVHNode* BVHNodes2Cuda = nullptr;
	// 根节点信息
	int* LBVHRootDataCuda = nullptr;

	// 转移到主存里的 结果数据
	double* distDataHost = nullptr;

	// 准备结果数组
	double* d_min_dists = nullptr;
	int* d_min_indices = nullptr;

#ifndef PROF
	// 对内存分配计时开始
	float start_Malloc_cpu = omp_get_wtime();
#endif

	// 开辟CUDA设备内存
	err = cudaMalloc((void**)&points1Cuda, size1);
	CUDA_CHECK_ERROR(err);

	err = cudaMalloc((void**)&points2Cuda, size2);
	CUDA_CHECK_ERROR(err);

	if (mode == CHECK_MODE_BRUTE_FORCE || mode == CHECK_MODE_TEST)
	{
		// 暴力模式下的内存分配
		err = cudaMalloc((void**)&distDataCuda, distSize);
		CUDA_CHECK_ERROR(err);

		distDataHost = (double*)malloc(distSize);
	}
	if (mode == CHECK_MODE_BVH || mode == CHECK_MODE_TEST) {
		// BVH模式下的内存分配
		// AABB内存分配
		err = cudaMalloc((void**)&aabbCuda1, 2 * sizeof(vec3f));
		CUDA_CHECK_ERROR(err);
		err = cudaMalloc((void**)&aabbCuda2, 2 * sizeof(vec3f));
		CUDA_CHECK_ERROR(err);

		// 莫顿码和索引内存分配，都要padding到2的幂次，以便排序
		// 计算填充后的大小
		int padded_num1 = nextPowerOf2(pointNumber1);
		int padded_num2 = nextPowerOf2(pointNumber2);
		// 莫顿码内存分配
		err = cudaMalloc((void**)&mortonCodes1Cuda, padded_num1 * sizeof(unsigned long long));
		CUDA_CHECK_ERROR(err);
		err = cudaMalloc((void**)&mortonCodes2Cuda, padded_num2 * sizeof(unsigned long long));
		CUDA_CHECK_ERROR(err);
		// 索引内存分配
		err = cudaMalloc((void**)&ObjectIDx1Cuda, padded_num1 * sizeof(int));
		CUDA_CHECK_ERROR(err);
		err = cudaMalloc((void**)&ObjectIDx2Cuda, padded_num2 * sizeof(int));
		CUDA_CHECK_ERROR(err);

		// LBVH节点内存分配
		err = cudaMalloc((void**)&BVHNodes1Cuda, (pointNumber1 * 2 - 1) * sizeof(LBVHNode));
		CUDA_CHECK_ERROR(err);

		err = cudaMalloc((void**)&BVHNodes2Cuda, (pointNumber2 * 2 - 1) * sizeof(LBVHNode));
		CUDA_CHECK_ERROR(err);

		// 根节点信息内存分配
		err = cudaMalloc((void**)&LBVHRootDataCuda, 4 * sizeof(int)); // 两对，一个存root idx，一个存root count
		CUDA_CHECK_ERROR(err);

		// 结果最小距离数组分配
		cudaMalloc((void**)&d_min_dists, pointNumber1 * sizeof(double));
		cudaMalloc((void**)&d_min_indices, pointNumber1 * sizeof(int));
	}


#ifndef PROF
	// 对数据传输计时开始
	double start_DataTransform_cpu = omp_get_wtime();
#endif
	
	// 复制数据到设备
	cudaMemcpy(points1Cuda, m1->_vtxs, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(points2Cuda, m2->_vtxs, size2, cudaMemcpyHostToDevice);

	if (mode == CHECK_MODE_BVH || mode == CHECK_MODE_TEST) {
		// BVH模式下，初始化根节点数据
		int initRootData[4] = { -1,0,-1,0 };
		cudaMemcpy(LBVHRootDataCuda, initRootData, 4 * sizeof(int), cudaMemcpyHostToDevice);
	}


#ifndef PROF
	// 对计算核函数计时开始
	double start_Kernel_cpu = omp_get_wtime();
#endif

	// 最后结果
	unsigned idx1_BF = 0, idx2_BF = 0;
	double minDist_BF = FLT_MAX;
	int idx1_BVH = 0, idx2_BVH = 0;
	double minDist_BVH = FLT_MAX;

	// 开始计算
	if (mode == CHECK_MODE_BRUTE_FORCE || mode == CHECK_MODE_TEST) {
		// 启动：暴力计算所有点对距离
		dim3 block(16, 16, 1);
		dim3 thread((pointNumber1 + block.x - 1) / block.x, (pointNumber2 + block.y - 1) / block.y, 1);
		BruteForce_CalculateDistance << < thread, block >> > (points1Cuda, points2Cuda, distDataCuda, pointNumber1, pointNumber2);

		// 将数据从设备复制回主机
		cudaMemcpy(distDataHost, distDataCuda, distSize, cudaMemcpyDeviceToHost);

		// 处理结果数据
		// 初始解法：直接计算所有点对距离，找出最小距离
		for (unsigned int i = 0; i < allPointsSize; i++) {
			if (distDataHost[i] < minDist_BF) {
				idx1_BF = i % pointNumber1;
				idx2_BF = i / pointNumber1;
				minDist_BF = distDataHost[i];
			}
		}
	}
	if (mode == CHECK_MODE_BVH || mode == CHECK_MODE_TEST) {
		// BVH模式下的距离计算
		compute_global_aabb_double_cuda(points1Cuda, pointNumber1, aabbCuda1, aabbCuda1 + 1);
		compute_global_aabb_double_cuda(points2Cuda, pointNumber2, aabbCuda2, aabbCuda2 + 1);

		// debug: 将AABB从设备复制回主机
		//vec3f aabbCpu1[2];
		//vec3f aabbCpu2[2];
		//cudaMemcpy(aabbCpu1, aabbCuda1, 2 * sizeof(vec3f), cudaMemcpyDeviceToHost);
		//cudaMemcpy(aabbCpu2, aabbCuda2, 2 * sizeof(vec3f), cudaMemcpyDeviceToHost);

		// 在计算morton code前，先对padding部分进行设置
		int padded_num1 = nextPowerOf2(pointNumber1);
		int padded_num2 = nextPowerOf2(pointNumber2);
		set_padding_cuda(mortonCodes1Cuda, ObjectIDx1Cuda, pointNumber1, padded_num1);
		set_padding_cuda(mortonCodes2Cuda, ObjectIDx2Cuda, pointNumber2, padded_num2);

		// 计算morton code
		compute_morton_codes_cuda(points1Cuda, pointNumber1,
			&(aabbCuda1[0]), &(aabbCuda1[1]),
			mortonCodes1Cuda,
			ObjectIDx1Cuda);
		compute_morton_codes_cuda(points2Cuda, pointNumber2,
			&(aabbCuda2[0]), &(aabbCuda2[1]),
			mortonCodes2Cuda,
			ObjectIDx2Cuda);


		// debug：将morton code从设备复制回主机（全部）
		//vector<unsigned long long> mc1cuda;
		//mc1cuda.resize(padded_num1);
		//cudaMemcpy(mc1cuda.data(), mortonCodes1Cuda, padded_num1 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		//vector<int> idx1cuda;
		//idx1cuda.resize(padded_num1);
		//cudaMemcpy(idx1cuda.data(), ObjectIDx1Cuda, padded_num1 * sizeof(int), cudaMemcpyDeviceToHost);

		//vector<unsigned long long> mc2cuda;
		//mc2cuda.resize(padded_num2);
		//cudaMemcpy(mc2cuda.data(), mortonCodes2Cuda, padded_num2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

		// 排序morton code和对应的index
		/*bitonic_sort_cuda(mortonCodes1Cuda, ObjectIDx1Cuda, nextPowerOf2(pointNumber1));
		bitonic_sort_cuda(mortonCodes2Cuda, ObjectIDx2Cuda, nextPowerOf2(pointNumber2));*/
		// 这部分总是有问题，放弃，直接当调包侠，使用thrust
		perform_thrust_sort(mortonCodes1Cuda, ObjectIDx1Cuda, padded_num1);
		perform_thrust_sort(mortonCodes2Cuda, ObjectIDx2Cuda, padded_num2);


		// debug：将morton code从设备复制回主机（全部）
		//cudaMemcpy(mc1cuda.data(), mortonCodes1Cuda, pointNumber1 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		//cudaMemcpy(idx1cuda.data(), ObjectIDx1Cuda, padded_num1 * sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMemcpy(mc2cuda.data(), mortonCodes2Cuda, pointNumber2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

		//cudaDeviceSynchronize();

		// debug CPU端验证其排序结果是否正确
		//std::cout << "查询点集1的Morton code排序情况 \n";
		//for (size_t i = 0, padErr = 0, sortErr = 0, dupWarn = 0; i < pointNumber1 - 1; i++) {
		//	unsigned long long thisMorton = mc1cuda[i];
		//	unsigned long long nextMorton = mc1cuda[i + 1];

		//	if (thisMorton == ULLONG_MAX) {
		//		std::cout << "Padding Morton code found at index " << i << "!\n";
		//		padErr++;
		//	}
		//	if (thisMorton > nextMorton) {
		//		std::cout << "Error: Morton codes not sorted at index " << i << "!\n";
		//		sortErr++;
		//	}
		//	if (thisMorton == nextMorton) {
		//		std::cout << "Warning: Duplicate Morton codes at index " << i << " and " << (i + 1) << "!\n";
		//		dupWarn++;
		//	}

		//	if (i == pointNumber1 - 2) {
		//		std::cout << "点集1结果：padding错误" << padErr << "个，排序错误" << sortErr << "个，重复警告" << dupWarn << "个。\n";
		//	}
		//}

		//std::cout << "查询点集2的Morton code排序情况：\n";
		//for (size_t i = 0, padErr = 0, sortErr = 0, dupWarn = 0; i < pointNumber2 - 1; i++) {
		//	unsigned long long thisMorton = mc2cuda[i];
		//	unsigned long long nextMorton = mc2cuda[i + 1];

		//	if (thisMorton == ULLONG_MAX) {
		//		std::cout << "Padding Morton code found at index " << i << "!\n";
		//		padErr++;
		//	}
		//	if (thisMorton > nextMorton) {
		//		std::cout << "Error: Morton codes not sorted at index " << i << "!\n";
		//		sortErr++;
		//	}
		//	if (thisMorton == nextMorton) {
		//		std::cout << "Warning: Duplicate Morton codes at index " << i << " and " << (i + 1) << "!\n";
		//		dupWarn++;
		//	}
		//	if (i == pointNumber2 - 2) {
		//		std::cout << "点集2结果：padding错误" << padErr << "个，排序错误" << sortErr << "个，重复警告" << dupWarn << "个。\n";
		//	}
		//}
		
		// 初始化BVH节点
		initialize_bvh_nodes(BVHNodes1Cuda, pointNumber1 * 2 - 1);
		initialize_bvh_nodes(BVHNodes2Cuda, pointNumber2 * 2 - 1);

		// 生成LBVH树
		build_lbvh_structure_cuda(
			mortonCodes1Cuda,
			ObjectIDx1Cuda,
			BVHNodes1Cuda,
			BVHNodes1Cuda + pointNumber1,
			pointNumber1);
		build_lbvh_structure_cuda(
			mortonCodes2Cuda,
			ObjectIDx2Cuda,
			BVHNodes2Cuda,
			BVHNodes2Cuda + pointNumber2,
			pointNumber2);

		// debug，将叶子数据从设备复制回主机，验证结构正确性
		//vector<LBVHNode> bvh1cpu;
		//bvh1cpu.resize(pointNumber1 * 2 - 1);
		//cudaMemcpy(bvh1cpu.data(), BVHNodes1Cuda, (pointNumber1 * 2 - 1) * sizeof(LBVHNode), cudaMemcpyDeviceToHost);

		//vector<LBVHNode> bvh2cpu;
		//bvh2cpu.resize(pointNumber2 * 2 - 1);
		//cudaMemcpy(bvh2cpu.data(), BVHNodes2Cuda, (pointNumber2 * 2 - 1) * sizeof(LBVHNode), cudaMemcpyDeviceToHost);

		// 动态查找根节点
		find_root(BVHNodes1Cuda + pointNumber1, pointNumber1, LBVHRootDataCuda, LBVHRootDataCuda + 1);
		find_root(BVHNodes2Cuda + pointNumber2, pointNumber2, LBVHRootDataCuda + 2, LBVHRootDataCuda + 3);

		// debug 将根节点数据从设备复制回主机
		//int rootDataHost[4];
		//cudaMemcpy(&rootDataHost, LBVHRootDataCuda, 4 * sizeof(int), cudaMemcpyDeviceToHost);

		// 自底向上，构建BVH每个节点的AABB
		refit_lbvh_structure_cuda(
			ObjectIDx1Cuda,
			points1Cuda,
			BVHNodes1Cuda,
			BVHNodes1Cuda + pointNumber1,
			pointNumber1);
		refit_lbvh_structure_cuda(
			ObjectIDx2Cuda,
			points2Cuda,
			BVHNodes2Cuda,
			BVHNodes2Cuda + pointNumber2,
			pointNumber2);

		// debug：将BVH节点（更新AABB后的）数据从设备复制回主机，验证结构正确性
		//cudaMemcpy(bvh1cpu.data(), BVHNodes1Cuda, (pointNumber1 * 2 - 1) * sizeof(LBVHNode), cudaMemcpyDeviceToHost);
		//cudaMemcpy(bvh2cpu.data(), BVHNodes2Cuda, (pointNumber2 * 2 - 1) * sizeof(LBVHNode), cudaMemcpyDeviceToHost);

		// debug：获取根节点AABB，和上文的全局AABB对比
		//vec3f rootAabbCpu1[2];
		//vec3f rootAabbCpu2[2];
		//cudaMemcpy(&rootAabbCpu1[0], &(BVHNodes1Cuda[rootDataHost[0]].min_box), sizeof(vec3f), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&rootAabbCpu1[1], &(BVHNodes1Cuda[rootDataHost[0]].max_box), sizeof(vec3f), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&rootAabbCpu2[0], &(BVHNodes2Cuda[rootDataHost[2]].min_box), sizeof(vec3f), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&rootAabbCpu2[1], &(BVHNodes2Cuda[rootDataHost[2]].max_box), sizeof(vec3f), cudaMemcpyDeviceToHost);

		// debug：验证AABB正确性
		//std::cout << "\nVerifying BVH 1 AABB...\n";
		//bool bvh1_ok = verify_bvh_aabb(bvh1cpu, pointNumber1);

		//std::cout << "\nVerifying BVH 2 AABB...\n";
		//bool bvh2_ok = verify_bvh_aabb(bvh2cpu, pointNumber2);

		//if (bvh1_ok && bvh2_ok) {
		//	std::cout << "\n>>> ALL CHECKS PASSED. Ready for Distance Query. <<<\n";
		//}
		//else {
		//	std::cout << "\n>>> FATAL: BVH Construction Failed. Do NOT proceed to Query. <<<\n";
		//}

		// 验证通过，开始查询距离

		// 3. 启动 Kernel: 点云 1 查 点云 2
		int threads = 256;
		int blocks = (pointNumber1 + threads - 1) / threads;

		query_distance_kernel << <blocks, threads >> > (
			points1Cuda,         // 查询点 (Tree 1 的原始点)
			pointNumber1,
			BVHNodes2Cuda,       // 目标树 (Tree 2)
			ObjectIDx2Cuda,      // Tree 2 的排序索引
			points2Cuda,         // Tree 2 的原始点 (备用)
			LBVHRootDataCuda + 2,// Tree 2 的根的索引
			pointNumber2,        // Tree 2 的叶子数
			d_min_dists,         // 输出
			d_min_indices
			);

		cudaDeviceSynchronize();

		// 查询最小的
		perform_thrust_min_element(d_min_dists, pointNumber1, &minDist_BVH, &idx1_BVH);

		// 从显存的特定位置拷贝 1 个 int 回来
		cudaMemcpy(&idx2_BVH,
			&d_min_indices[idx1_BVH], // 地址偏移
			sizeof(int),
			cudaMemcpyDeviceToHost);

		// 输出结果
		//std::cout << ">>> Pair Found: Cloud1[" << best_query_idx << "] <-> Cloud2[" << best_target_idx << "]" << std::endl;
		//std::cout << ">>> Distance: " << global_min_dist << std::endl;

	}

	// 准备结果
	if (mode == CHECK_MODE_BRUTE_FORCE) {
		rets.clear();
		rets.push_back(id_pair(idx1_BF, idx2_BF, false));
	}
	if (mode == CHECK_MODE_BVH) {

		rets.clear();
		rets.push_back(id_pair(idx1_BVH, idx2_BVH, false));
	}

	if (mode == CHECK_MODE_TEST) {
		// 同时运行两种模式，比较结果
		rets.clear();
		rets.push_back(id_pair(idx1_BF, idx2_BF, false));
		rets.push_back(id_pair(idx1_BVH, idx2_BVH, false));
		// 输出对比信息
		std::cout << "\n=== CHECK MODE: BRUTE FORCE VS BVH ===\n";
		std::cout << "Brute Force Result: Cloud1[" << idx1_BF << "] <-> Cloud2[" << idx2_BF << "], Distance = " << minDist_BF << "\n";
		std::cout << "BVH Result        : Cloud1[" << idx1_BVH << "] <-> Cloud2[" << idx2_BVH << "], Distance = " << minDist_BVH << "\n";
		if (idx1_BF == idx1_BVH && idx2_BF == idx2_BVH) {
			std::cout << ">>> CHECK PASSED: Both methods found the SAME closest pair. <<<\n\n";
		}
		else {
			std::cout << ">>> CHECK FAILED: Methods found DIFFERENT closest pairs! <<<\n\n";
		}
	}

#ifndef PROF
	// 对计算核函数计时结束
	double stop_Kernel_cpu = omp_get_wtime();
#endif

	// 释放CUDA设备内存
	cudaFree(points1Cuda);
	cudaFree(points2Cuda);
	if (mode == CHECK_MODE_BRUTE_FORCE || mode == CHECK_MODE_TEST) {
		cudaFree(distDataCuda);

		// 释放主机内存
		free(distDataHost);
	}
	if (mode == CHECK_MODE_BVH || mode == CHECK_MODE_TEST) {
		cudaFree(aabbCuda1);
		cudaFree(aabbCuda2);  

		cudaFree(mortonCodes1Cuda);
		cudaFree(mortonCodes2Cuda);
		cudaFree(ObjectIDx1Cuda);
		cudaFree(ObjectIDx2Cuda);

		cudaFree(BVHNodes1Cuda);
		cudaFree(BVHNodes2Cuda);
		cudaFree(LBVHRootDataCuda);

		cudaFree(d_min_dists);
		cudaFree(d_min_indices);
	}


#ifndef PROF
	float all_stop_time_cpu = omp_get_wtime();
#endif

#ifndef PROF
	std::cout << "\nPrepare Time: " << (start_Malloc_cpu - all_start_time_cpu) * 1000.0 << " ms\n";
	std::cout << "CPU Malloc Time: " << (start_DataTransform_cpu - start_Malloc_cpu) * 1000.0 << " ms\n";
	std::cout << "CPU Data Transform Time: " << (start_Kernel_cpu - start_DataTransform_cpu) * 1000.0 << " ms\n";
	std::cout << "CPU Kernel Time: " << (stop_Kernel_cpu - start_Kernel_cpu) * 1000.0 << " ms\n";
	std::cout << "CPU Free Time: " << (all_stop_time_cpu - stop_Kernel_cpu) * 1000.0 << " ms\n";
	std::cout << "\nTotal CUDA Time: " << (all_stop_time_cpu - all_start_time_cpu) * 1000.0 << " ms\n\n";
#endif

	return minDist_BF;
}