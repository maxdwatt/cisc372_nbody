#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
#include <cuda.h>
#define BLOCK_THREAD 16

__global__ void accelPair(vector3** accels,vector3* values, vector3* hPos, double* mass){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(i = 0; i < NUMENTITIES; i++){
    if( i < NUMENTITIES){
	for(int j = 0; j < NUMENTITIES; j++){
	    if(i == j){
		FILL_VECTOR(accels[i][j],0,0,0);
	    }
	    else{
		vector3 distance;
		for (int k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
                double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                double magnitude=sqrt(magnitude_sq);
                double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
                 FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);                      
		}
	    }
	}
    }
    //_syncthreads();
}

/*__global__ void sumRows(vector3** accels, vector3* values){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (i=0;i<NUMENTITIES;i++){
            vector3 accel_sum={0,0,0};
            for (j=0;j<NUMENTITIES;j++){
            	for (k=0;k<3;k++)
                    accel_sum[k]+=accels[i][j][k];
                }
                //compute the new velocity based on the acceleration and time interval
                //compute the new position based on the velocity and time interval
                for (k=0;k<3;k++){
                        hVel[i][k]+=accel_sum[k]*INTERVAL;
                        hPos[i][k]=hVel[i][k]*INTERVAL;
                }
        }
}*/
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
        vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
        vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	vector3** d_accels;
	vector3* d_values;
	//cpu activating storage on the GPU
	cudaMalloc((void**)d_accels, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**)&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	//cpu copies data from itself to the GPU
	cudaMemcpy(d_accels,accels,sizeof(vector3*)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_values,values,sizeof(vector3*)*NUMENTITIES*NUMENTITIES, cudaMemcpyDeviceToHost);
	//launching te kernal on gpu
	int blocks = (NUMENTITIES + BLOCK_THREAD - 1) / BLOCK_THREAD;
	accelPair<<<BLOCK_THREAD,blocks>>>(d_accels,d_values,hPos,mass);
	//copying results back to the cpu
	cudaMemcpy(accels,d_accels,sizeof(vector3*)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(values,d_values,sizeof(vector3*)*NUMENTITIES*NUMENTITIES, cudaMemcpyDeviceToHost);
	/*for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}*/
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	/*sumRows<<<BLOCK_THREAD,blocks>>>(d_accels,d_values);
        cudaMemcpy(accels,d_accels,sizeof(Vector3*)*NUMENTITIES, cudaMemcpyDeviceToHost);
        cudaMemcpy(values,d_values,sizeof(Vector3*)*NUMENTITIES*NUMENTITIES, cudaMemcpyDeviceToHost);*/
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}
	}
	cudaFree(d_accels);
        cudaFree(d_values);
	free(accels);
	free(values);
}
