#include <cuda.h>
#include <stdio.h>

#define NUMENTITIES 1000

typedef double vector3[3];

__global__ void sumAccelerations(vector3** accels, vector3* accelSum, vector3* hVel, vector3* hPos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUMENTITIES) {
        for (int j = 0; j < NUMENTITIES; j++) {
            for (int k = 0; k < 3; k++) {
               accelSum[i][k] += 5;
            }
        }
	for (int b=0;b<3;b++){
		hVel[i][b] += accelSum[i][b];
		hPos[i][b] += accelSum[i][b];
	}
    }
}

int main() {
    vector3* values = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    vector3** accels = (vector3**)malloc(sizeof(vector3*) * NUMENTITIES);
    for (int i = 0; i < NUMENTITIES; i++)
        accels[i] = &values[i * NUMENTITIES];

    // Allocate memory on the GPU
    vector3* d_values;
    vector3** d_accels;
    cudaMalloc((void**)&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    cudaMalloc((void**)&d_accels, sizeof(vector3*) * NUMENTITIES);

    // Copy data from host to device
    cudaMemcpy(d_values, values, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accels, accels, sizeof(vector3*) * NUMENTITIES, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    int blockSize = 256;
    int gridSize = (NUMENTITIES + blockSize - 1) / blockSize;

    // Create a vector3 array on the host to store the sum of accelerations
    vector3* h_accelSum = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);
    vector3* h_Pos = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);
    vector3* h_Vel = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);
    // Create a vector3 array on the device to store the sum of accelerations
    vector3* d_accelSum;
    vector3* d_Pos;
    vector3* d_Vel;
    cudaMalloc((void**)&d_accelSum, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_Pos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_Vel, sizeof(vector3) * NUMENTITIES);

    // Launch the CUDA kernel
    sumAccelerations<<<gridSize, blockSize>>>(d_accels, d_accelSum,d_Pos,d_Vel);

    // Copy the result back from the device to the host
    cudaMemcpy(h_accelSum, d_accelSum, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Pos, d_Pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Vel, d_Vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    // Print the result
    for (int i = 0; i < NUMENTITIES; i++) {
        printf("Accel sum for entity %d: (%f, %f, %f), hPos is: %F, hVel is %f\n", i, h_accelSum[i][0], h_accelSum[i][1], h_accelSum[i][2],*(h_Vel[i]),*(h_Pos[i]));
    }

    // Free the allocated memory
    free(values);
    free(accels);
    free(h_accelSum);
    cudaFree(d_values);
    cudaFree(d_accels);
    cudaFree(d_accelSum);

    return 0;
}
