#include "checkCuda.h"

__device__ void testCudaFunction() {
	// This function does nothing, it's just for checking CUDA setup
}

__global__ void CalculateDistance(vec3f* points1, vec3f* points2, double* distData, size_t pointNumber1, size_t pointNumber2) {
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

REAL checkDistCuda(const kmesh* m1, const kmesh* m2, std::vector<id_pair>& rets) {
	// 报错信息
	cudaError_t err = cudaSuccess;

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
	vec3f* points1Cuda = nullptr;
	vec3f* points2Cuda = nullptr;
	double* distDataCuda = nullptr;
	unsigned int* ansDataCuda = nullptr;

	// 结果数据
	double* distDataHost = (double*)malloc(distSize);

	// 将顶点信息复制到CUDA设备内存
	err = cudaMalloc((void**)&points1Cuda, size1);
	if (err != cudaSuccess) {
		std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
		return -1;
	}
	err = cudaMalloc((void**)&points2Cuda, size2);
	if (err != cudaSuccess) {
		std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
		return -1;
	}
	err = cudaMalloc((void**)&distDataCuda, distSize);
	if (err != cudaSuccess) {
		std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
		return -1;
	}

	// 复制数据到设备
	cudaMemcpy(points1Cuda, m1->_vtxs, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(points2Cuda, m2->_vtxs, size2, cudaMemcpyHostToDevice);

	// 启动一个简单的CUDA内核来测试CUDA环境
	dim3 block(16, 16, 1);
	dim3 thread((size1 + block.x - 1) / block.x, (size2 + block.y - 1) / block.y, 1);
	CalculateDistance<<< thread, block>>> (points1Cuda, points2Cuda, distDataCuda, pointNumber1, pointNumber2);

	// 将数据从设备复制回主机
	cudaMemcpy(distDataHost, distDataCuda, distSize, cudaMemcpyDeviceToHost);

	// 处理结果数据
	// 初始解法：直接计算所有点对距离，找出最小距离
	unsigned idx1, idx2;
	double minDist = FLT_MAX;
	for (unsigned int i = 0; i < allPointsSize; i++) {
		if (distDataHost[i] < minDist) {
			idx1 = i % pointNumber1;
			idx2 = i / pointNumber1;
			minDist = distDataHost[i];
		}
	}
	rets.clear();
	rets.push_back(id_pair(idx1, idx2, false));

	// 释放CUDA设备内存
	cudaFree(points1Cuda);
	cudaFree(points2Cuda);
	cudaFree(distDataCuda);

	return minDist;
}