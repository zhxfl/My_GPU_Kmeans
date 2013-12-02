/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_kmeans.cu  (CUDA version)                            */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// -----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
const int inf = 1 << 30;
#define BLOCK_SIZE 16

void write(char * filename, int n, int m, float** matrix) {
	FILE* file = fopen(filename, "w");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			fprintf(file, "%f ", matrix[i][j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}

__global__ void cal_dis(float* A, float*B, float *C, int n1, int m1, int n2,
		int m2, int n3, int m3) {
	int by = blockIdx.x;
	int bx = blockIdx.y;
	int ty = threadIdx.x;
	int tx = threadIdx.y;

	int startA = bx * m1 * BLOCK_SIZE;
	int endA = startA + m1 - 1;
	int stepA = BLOCK_SIZE;

	int startB = BLOCK_SIZE * by;
	int stepB = m2 * BLOCK_SIZE;

	float Csub = 0; // every thread has a Csub
	int t1 = m1 * tx + ty;
	int t2 = m2 * tx + ty;

	//printf("%d %d\n",tx,ty);
	for (int a = startA, b = startB; a <= endA; a += stepA, b += stepB) {
		__shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

		if (BLOCK_SIZE * bx + tx < n1 && a - startA + ty < m1)
			Asub[tx][ty] = A[a + t1];
		else {
			Asub[tx][ty] = 0;
			//printf("A %d %d %d %d %d %d\n",bx, by, tx,ty,BLOCK_SIZE * bx + tx, a - startA + ty);
		}

		if ((b - startB) / m2 + tx < n2 && BLOCK_SIZE * by + ty < m2) {
			Bsub[tx][ty] = B[b + t2];
			//printf("=B %d %d %d %d %d %d\n",bx, by, tx,ty,( b - startB ) / m2 + tx, BLOCK_SIZE * by + ty);
			//__syncthreads();
		} else {
			Bsub[tx][ty] = 0;
			//printf("B %d %d %d %d %d %d\n",bx, by, tx,ty,( b - startB ) / m2 + tx, BLOCK_SIZE * by + ty);
		}

		//printf("%f %f\n", Asub[tx][ty], Bsub[tx][ty]);
		__syncthreads();

#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			float t = (Asub[tx][k] - Bsub[k][ty]);
			Csub += t * t;
		}

		__syncthreads();
		//printf("tx = %d , k = %d : %f\n", tx, BLOCK_SIZE - 1,x1);
		//printf("k = %d , ty = %d : %f\n", BLOCK_SIZE - 1, ty,y1);
	}
	//printf("yes");
	int x = BLOCK_SIZE * bx + tx;
	int y = BLOCK_SIZE * by + ty;
	if (x < n3 && y < m3) {
		C[x + y * n3] = Csub;
		//printf("C[%d][%d] = %f\n",Csub);
	}
}

__global__ void get_min_id(float * dis, int n3, int m3, int *id, int *delta) {
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int tId = bx * blockDim.x + tx;
	if (tId >= n3)
		return;
	int start = tId;
	float dis_ = dis[start];
	int id_ = 0;
	for (int i = 1; i < m3; i++) {
		start += n3;
		float t = dis[start];
		if (dis_ > t) {
			dis_ = t;
			id_ = i;
		}
	}

	delta[tId] = 0;

	if (id_ != id[tId]) {
		delta[tId] = 1;
		id[tId] = id_;
	}
	__syncthreads();

	start = bx * blockDim.x;
	for (int i = (blockDim.x + 1) >> 1; i >= 1; i >>= 1) {
		if (start + tx + i < n3) {
			delta[start + tx] += delta[start + tx + i];
		}
		__syncthreads();
	}
}

//TODO
//cluster[d][k]
//objs[n][d];

__global__ void get_cluster_size(float * cluster, int numCoords,
		int numClusters, int numObjs, int * clustersize, int *id, int bx) {
	int tx = threadIdx.x;
	int tId = bx * blockDim.x + tx;
	if (tId >= numObjs)
		return;
	int index = id[tId];
	atomicAdd(&clustersize[index], 1);
	if (clustersize[index] >= 1) {
		for (int j = 0; j < numCoords; j++) {
			cluster[index + j * numClusters] = 0;
		}
	}
}

__global__ void get_new_cluster_1(float *cluster, int n1, int m1, float*objs,
		int n2, int m2, int * clustersize, int *id, int bx) {
	int tx = threadIdx.x;
	int tId = bx * blockDim.x + tx;
	if (tId >= n2)
		return;
	int index = id[tId];

	for (int j = 0; j < n1; j++) {
		if (clustersize[index] > 0) {
			atomicAdd(&cluster[index + j * m1], objs[tId * m2 + j]);
		}
	}
}

__global__ void get_new_cluster_2(float *cluster, int n1, int m1,
		int * clustersize) {
	/*for (int i = 0; i < numClusters  ; i++) {
	 for (int j = 0; j < numCoords; j++) {
	 if (newClusterSize[i] > 0)
	 dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
	 newClusters[j][i] = 0.0;
	 }
	 newClusterSize[i] = 0;
	 }*/
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int tId = bx * blockDim.x + tx;
	if (tId >= m1)
		return;
	int size = clustersize[tId];
	if (size > 0)
		for (int j = 0; j < n1; j++) {
			float t = cluster[j * m1 + tId];
			cluster[j * m1 + tId] = t / size;
		}
}

/*----< cuda_kmeans() >-------------------------------------------------------*/
//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
float** cuda_kmeans(float **objects, /* in: [numObjs][numCoords] */
int numCoords, /* no. features */
int numObjs, /* no. objects */
int numClusters, /* no. clusters */
float threshold, /* % objects change membership */
int *membership, /* out: [numObjs] */
int *loop_iterations) {
	int i, j, index, loop = 0;
	int *newClusterSize; /* [numClusters]: no. objects assigned in each
	 new cluster */
	float delta; /* % of objects change their clusters */
	float **dimObjects;
	float **clusters; /* out: [numClusters][numCoords] */
	float **dimClusters;
	float **newClusters; /* [numCoords][numClusters] */
	float **dimdis;
	int *dimMinId;
	int *hostDetal;
	int *hostClustersSize;

	float *deviceObjects;
	float *deviceClusters;
	int *deviceMinId;
	int *deviceDelta;
	int *deviceClusterSize;
	float *devicedis;

	dimMinId = (int*) malloc(sizeof(int) * numObjs);

	malloc2D(dimObjects, numObjs, numCoords, float);
	for (i = 0; i < numObjs; i++) {
		for (j = 0; j < numCoords; j++) {
			dimObjects[i][j] = objects[i][j];
		}
	}
	malloc2D(dimClusters, numCoords, numClusters, float);

	for (i = 0; i < numClusters; i++) {
		for (j = 0; j < numCoords; j++) {
			dimClusters[j][i] = dimObjects[i][j];
		}
	}
	malloc2D(dimdis, numObjs, numClusters, float);
	for (i = 0; i < numObjs; i++) {
		for (j = 0; j < numClusters; j++) {
			dimdis[i][j] = inf;
		}
	}

	for (i = 0; i < numObjs; i++)
		dimMinId[i] = -1;

	newClusterSize = (int*) calloc(numClusters, sizeof(int));
	assert(newClusterSize != NULL);

	malloc2D(newClusters, numCoords, numClusters, float);
	memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);

	checkCuda(cudaMalloc(&deviceObjects, numObjs * numCoords * sizeof(float)));
	checkCuda(
			cudaMalloc(&deviceClusters,
					numClusters * numCoords * sizeof(float)));
	checkCuda(cudaMalloc(&devicedis, numClusters * numObjs * sizeof(float)));
	checkCuda(cudaMalloc(&deviceMinId, numObjs * sizeof(int)));
	checkCuda(cudaMalloc(&deviceClusterSize, numObjs * sizeof(int)));

	checkCuda(
			cudaMemcpy(deviceObjects, dimObjects[0],
					numObjs * numCoords * sizeof(float),
					cudaMemcpyHostToDevice));
	checkCuda(
			cudaMemcpy(devicedis, dimdis[0],
					numClusters * numObjs * sizeof(float),
					cudaMemcpyHostToDevice));

	dim3 blocks1 = dim3((numClusters + BLOCK_SIZE - 1) / BLOCK_SIZE,
			(numObjs + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 threads1 = dim3(BLOCK_SIZE, BLOCK_SIZE);

	dim3 blocks2 = dim3(
			numObjs >= deviceProp.maxThreadsPerBlock ?
					((numObjs + deviceProp.maxThreadsPerBlock - 1) / 1024) : 1);
	dim3 threads2 = dim3(
			numObjs >= deviceProp.maxThreadsPerBlock ? 1024 : numObjs);

	dim3 blocks3 = dim3(
			numClusters >= deviceProp.maxThreadsPerBlock ?
					((numClusters + deviceProp.maxThreadsPerBlock - 1) / 1024) :
					1);
	dim3 threads3 = dim3(
			numClusters >= deviceProp.maxThreadsPerBlock ? 1024 : numClusters);

	checkCuda(cudaMalloc(&deviceDelta, sizeof(int) * numObjs));
	hostDetal = (int*) malloc(sizeof(int) * numObjs);

	hostClustersSize = (int*) malloc(sizeof(int) * numClusters);
	memset(hostClustersSize, 0, sizeof(int) * numClusters);

	checkCuda(
			cudaMemcpy(deviceClusters, dimClusters[0],
					numClusters * numCoords * sizeof(float),
					cudaMemcpyHostToDevice));
	checkCuda(
			cudaMemcpy(deviceMinId, dimMinId, numObjs * sizeof(float),
					cudaMemcpyHostToDevice));
	do {
		checkCuda(
				cudaMemcpy(deviceClusters, dimClusters[0],
						numClusters * numCoords * sizeof(float),
						cudaMemcpyHostToDevice));

		cal_dis<<<blocks1, threads1>>>(deviceObjects, deviceClusters, devicedis,
				numObjs, numCoords, numCoords, numClusters, numObjs,
				numClusters);

		cudaDeviceSynchronize();
		checkLastCudaError();

		get_min_id<<<blocks2, threads2>>>(devicedis, numObjs, numClusters,
				deviceMinId, deviceDelta);
		cudaDeviceSynchronize();
		checkLastCudaError();

		checkCuda(
				cudaMemcpy(dimMinId, deviceMinId, sizeof(float) * numObjs,
						cudaMemcpyDeviceToHost));
		checkCuda(
				cudaMemcpy(hostDetal, deviceDelta, sizeof(int) * numObjs,
						cudaMemcpyDeviceToHost));
		delta = 0;
		for (int i = 0; i < blocks2.x; i++) {
			delta += hostDetal[i * threads2.x];
		}

		for (int i = 0; i < numObjs; i++) {
			index = dimMinId[i];
			newClusterSize[index]++;
			for (int j = 0; j < numCoords; j++) {
				newClusters[j][index] += objects[i][j];
			}
		}

		for (int i = 0; i < numClusters; i++) {
			for (int j = 0; j < numCoords; j++) {
				if (newClusterSize[i] > 0)
					dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
				newClusters[j][i] = 0.0;
			}
			newClusterSize[i] = 0;
		}

		delta /= numObjs;
	} while (delta > threshold && loop++ < 512);

	malloc2D(clusters, numClusters, numCoords, float);
	for (i = 0; i < numClusters; i++) {
		for (j = 0; j < numCoords; j++) {
			clusters[i][j] = dimClusters[j][i];
		}
	}

	*loop_iterations = loop + 1;
	membership = dimMinId;
	free(dimObjects[0]);
	free(dimObjects);
	free(dimClusters[0]);
	free(dimClusters);
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);

	free(dimdis);
	free(dimdis[0]);
	free(dimMinId);
	free(hostDetal);
	free(hostClustersSize);

	checkCuda(cudaFree(deviceObjects));
	checkCuda(cudaFree(deviceClusters));
	checkCuda(cudaFree(deviceMinId));

	checkCuda(cudaFree(deviceDelta));
	checkCuda(cudaFree(deviceClusterSize));
	checkCuda(cudaFree(devicedis));
	return clusters;
}
