#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_THREAD 256

// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
vector3 *d_values, *d_sum;
vector3** d_accels;
double *mass, *d_mass;

//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
}

//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j, c = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

__global__ void accelMatrix(vector3** accels, vector3* values){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if( i < NUMENTITIES){
                accels[i] = &values[i * NUMENTITIES];
        }

}

__global__ void accelPair(vector3** accels,vector3* values, vector3* hPos, double* mass){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if( i < NUMENTITIES){
        if(j < NUMENTITIES){
            if(i == j){
                FILL_VECTOR(accels[i][j],0,0,0);
            }
            else{
                //FILL_VECTOR(accels[i][j],1,1,1);
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

__global__ void sumAccelerations(vector3** accels, vector3* accelSum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUMENTITIES) {
        for (int j = 0; j < NUMENTITIES; j++) {
            for (int k = 0; k < 3; k++) {
               accelSum[i][k] += accels[i][j][k];
            }
        }
    }
}

__global__ void updateh(vector3* pos, vector3* vel, vector3* accelSum){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < NUMENTITIES){
        	for(int k = 0; k < 3; k++){
                	vel[i][k] += accelSum[i][k] + INTERVAL;
                	pos[i][k] += accelSum[i][k] + INTERVAL;
        	}
	}
}

void initDeviceMemory(){
	cudaMalloc(&d_accels,sizeof(vector3*)*NUMENTITIES);
	cudaMalloc(&d_values,sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc(&d_mass,sizeof(double) * NUMENTITIES);
	cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc(&d_sum,sizeof(vector3) * NUMENTITIES);
	cudaMalloc(&d_hVel,sizeof(vector3) * NUMENTITIES);
}

void freeDeviceMemory(){
	cudaFree(d_accels);
	cudaFree(d_values);
	cudaFree(d_mass);
	cudaFree(d_hPos);
	cudaFree(d_sum);
	cudaFree(d_hVel);
}
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	//vector3** d_accels;
	//vector3* d_values;
	//cudaMalloc(&d_accels,sizeof(vector3*)*NUMENTITIES);
	//cudaMalloc(&d_values,sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	int blocks = (NUMENTITIES * NUMENTITIES + BLOCK_THREAD - 1)/ BLOCK_THREAD;
	accelMatrix<<<blocks,BLOCK_THREAD>>>(d_accels,d_values);
	//first compute the pairwise accelerations.  Effect is on the first argument.
	//putting mass on the device for use.
	//cudaMalloc(&d_mass,sizeof(double) * NUMENTITIES);
        cudaMemcpy(d_mass,mass,sizeof(double) * NUMENTITIES,cudaMemcpyHostToDevice);
	//getting d_hPos on the device for use 
	//cudaMalloc(&d_hPos, sizeof(vector3) * NUMENTITIES);
	cudaMemcpy(&d_hPos,hPos,sizeof(vector3) * NUMENTITIES,cudaMemcpyHostToDevice);
	//call the kernel for computing the acceleration
	accelPair<<<blocks,BLOCK_THREAD>>>(d_accels,d_values,d_hPos,d_mass);
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	//creating accelSum array on the gpu
	//vector3* d_sum;
	//cudaMalloc(&d_sum,sizeof(vector3) * NUMENTITIES);
	//launching the sum kernel
	sumAccelerations<<<blocks,BLOCK_THREAD>>>(d_accels,d_sum);
	//putting hVel on the gpu
	//cudaMalloc(&d_hVel,sizeof(vector3) * NUMENTITIES);
	cudaMemcpy(&d_hVel,hVel,sizeof(vector3) * NUMENTITIES,cudaMemcpyHostToDevice);
	//calling the kernel to update the values
	updateh<<<blocks,BLOCK_THREAD>>>(d_hPos,d_hVel,d_sum);
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);
	initDeviceMemory();
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	#endif
	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute();
	}
	clock_t t1=clock()-t0;
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
	freeDeviceMemory();
}
