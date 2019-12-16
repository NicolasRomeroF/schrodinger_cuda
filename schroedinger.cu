#include <stdio.h>

__global__ void schroedinger(float * H2, float * H1, float * H0, float c, float dt, float dd){
	int blocksize = blockDim.y*blockDim.x;
	int blockId = gridDim.x*blockIdx.y + blockIdx.x;
	int tid = blockId*blocksize + blockDim.x*threadIdx.y + threadIdx.x;

	int tidDown = tid + blockDim.x;
	int tidUp = tid - blockDim.x;
	int tidRight = tid + blockDim.y;
	int tidLeft = tid - blockDim.y;
	H2[tid] = 2*H1[tid] - 2*H0[tid] + c*c*(dt/dd)*(dt/dd)*(H1[tidDown] + H1[tidUp] + H1[tidLeft] + H1[tidRight] - 4*H1[tid]);

}


__host__ int main(){
	dim3 blocksize;
	dim3 gridsize;

	float c = 1.0;
	float dt = 0.1;
	float dd = 2.0;


	int t = 300;
	int x = 256;
	int y = 256;
	float * H0 = (float*)malloc(sizeof(float*)*y*x);

	float * H1 = (float*)malloc(sizeof(float*)*y*x);
	
	float * H2 = (float*)malloc(sizeof(float*)*y*x);

	float * h0,* h1,* h2;
	cudaMalloc(&h0, x*y*sizeof(float));
	cudaMalloc(&h1, x*y*sizeof(float));
	cudaMalloc(&h2, x*y*sizeof(float));

	cudaMemcpy(h0,H0, x*y*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(h1,H1, x*y*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(h2,H2, x*y*sizeof(float), cudaMemcpyHostToDevice);

	gridsize.x = x;
	gridsize.y = y;
	blocksize.x = 32;
	blocksize.y = 32;

	schroedinger<<<gridsize,blocksize>>>(h2,h1,h0,c,dt,dd);

	cudaMemcpy(H2, h2, x*y*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%f ",h2[2]);

	return 0;
}