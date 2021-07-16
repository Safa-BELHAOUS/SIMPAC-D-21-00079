
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

// global variables

#define dt_1  0.01f // time step in seconds
#define dx_1  0.001f // space step in m


#define  NX  51      // number of node points in X direction
#define  NY  51      // number of node points in X direction
#define  CENTRE 51/2

#define MAX_ITER 10000  // max number of iterations

#define K_1 43 // thermal conductivity
#define RHO 7800 // density
#define C_P 473 // specific heat 
#define T_SOURCE 30 // thermal source

#define alpha  ((dt_1 * K_1) / (RHO * C_P * dx_1 * dx_1))

template <class T> __device__ __host__ void swap(T& a, T& b) {
	T c(a); a = b; b = c;

}

// initialization

void init_temp(float* temp)
{
	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			int index = i + j * NX;
			if (i == CENTRE && j == CENTRE)
				temp[index] = T_SOURCE;
			else
				temp[index] = 0.0;
		}
	}

}

// parallel heat equation GPU Kernel
__global__ void temperature_update_kernelV1(float* T_old, float* T_new)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	bool compute_if = i > 0 && i < (NX - 1) && j>0 && j < (NY - 1);
 
	int offset = j * (NX)+i;           // point (i,j)              
	int up = (j + 1) * (NX)+i;       // point (i,j+1)            
	int down = (j - 1) * (NX)+i;       // point (i,j-1)    
	int right = j * (NX)+(i + 1);       // point (i+1,j)            
	int left = j * (NX)+(i - 1);       // point (i-1,j)            


	// fix the heat source to 30
	if (i == CENTRE && j == CENTRE) {
		T_new[offset] = T_SOURCE;
	}
	else if (compute_if) {

		T_new[offset] = T_old[offset]
			+ alpha * (T_old[right] + T_old[left] + T_old[up] + T_old[down] - 4 * T_old[offset]);
	}
}
// sequential  heat equation verion
void temperature_update_FDM(float* temp_1, float* temp_2) {

	int i, j, q, t;

	for (t = 0; t < MAX_ITER; t++) {

		for (i = 1; i < NX - 1; i++) {
			for (j = 1; j < NY - 1; j++) {

				int offset = j * (NX)+i;
				int left = j * (NX)+(i - 1);
				int right = j * (NX)+(i + 1);
				int down = (j - 1) * (NX)+i;
				int up = (j + 1) * (NX)+i;

				if (i == CENTRE && j == CENTRE) {

					temp_2[offset] = T_SOURCE;

				}
				else {
					temp_2[offset] = temp_1[offset]
						+ alpha * (temp_1[right] + temp_1[left] + temp_1[up] + temp_1[down] - 4 * temp_1[offset]);
				}
			}
		}


		swap(temp_1, temp_2);

	}

}


void savetemp(float* temp) {
	int i, j, i2d;
	FILE* fp;
	fp = fopen("temp.txt", "wb");
	for (j = NX - 1; j >= 0; j--) {
		for (i = 0; i < NY; i++) {
			i2d = j * NX + i;
			fprintf(fp, "%f ", temp[i2d]);
		}
		fprintf(fp, "%\n");
	}

	fclose(fp);
}

int main(int argc, char** argv)
{
	float* _temp1, * _temp2;  // pointers to device (GPU) memory

	// allocate a temperature array on the host
	float* temp_1 = new float[NX * NY];
	float* temp_2 = new float[NX * NY];
	// initialize array on the host
	init_temp(temp_1);
	init_temp(temp_2);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// allocate storage space on the GPU
	cudaMalloc((void**)&_temp1, NX * NY * sizeof(float));
	cudaMalloc((void**)&_temp2, NX * NY * sizeof(float));

	// copy (initialized) host arrays to the GPU memory from CPU memory
	cudaMemcpy(_temp1, temp_1, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(_temp2, temp_1, NX * NY * sizeof(float), cudaMemcpyHostToDevice);

	// assign a 2D distribution of CUDA "threads" within each CUDA "block"    
	int ThreadsPerBlock = 16;
	dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock);

	// calculate number of blocks along X and Y in a 2D CUDA "grid"
	dim3 dimGrid(ceil(float(NX) / float(dimBlock.x)), ceil(float(NY) / float(dimBlock.y)), 1);

	// begin Jacobi iteration
	int k = 0;
	while (k<MAX_ITER) {
		temperature_update_kernelV1 << <dimGrid, dimBlock >> >(_temp1, _temp2);   // update T1 using data stored in T2
		temperature_update_kernelV1 << <dimGrid, dimBlock >> >(_temp2, _temp1);   // update T2 using data stored in T1
		k += 2;
	}
	printf("it : %d\n", k);
	// copy final array to the CPU from the GPU
	cudaMemcpy(temp_1, _temp2, NX*NY*sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	printf("it : %d\n", k);
	// copy final array to the CPU from the GPU 
	cudaMemcpy(temp_1, _temp1, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << milliseconds * 0.001 << " ";
	std::cout << std::endl;

	/*int k = 0;
	while (k < MAX_ITER) {
		temperature_update_FDM(temp_1, temp_2);
		k++;
	}*/

	savetemp(temp_1);

	delete temp_1;
	delete temp_2;
	// release memory on the GPU
	cudaFree(_temp1);
	cudaFree(_temp2);

	return 0;
}
