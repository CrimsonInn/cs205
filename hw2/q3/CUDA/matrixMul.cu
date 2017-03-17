#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define BLOCK_SIZE 16

void printDeviceProp(const cudaDeviceProp &prop) {
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %lu.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %lu.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %lu.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %lu.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %lu.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

bool InitCUDA() {
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0)  {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    int i;
    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printDeviceProp(prop);
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
    cudaSetDevice(i);
    return true;
}


//CPU
void matrixMulCPU(float* res,const float *matrixA,const float *matrixB,int colsA,int rowsA,int rowsB) {
    float sum = 0;
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < rowsB; ++j) {
            sum = 0;
            for (int k = 0; k < colsA; ++k) {
                sum += (float)matrixA[i*colsA+k]*(float)matrixB[k*rowsB+ j];
            }
            res[i*rowsB+j] = (float)sum;
        }
    }
}

// GPU
// C(i,j) = sum{A(i, k)* B(k ,j)}
// each thread cal C(i, j)
__global__ void matrixMulGPUKernal0(float* matrixC,const float* matrixA,const float *matrixB,int colsA,int rowsB) {
    
    float sum = 0;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    for (int i = 0; i < colsA; ++i) {
        sum += matrixA[row*colsA + i] * matrixB[i*rowsB + col];
    }
    matrixC[row*rowsB + col] = sum;
}

// Csub(i,j) = sum{A(i,ksub+offsetA)*B(ksub+offsetB,j)}  0 <= ksub < blockSize
// C(i,j) = sum{Csub(i,j)}
// each thread cal each block
__global__ void matrixMulGPUKernal1(float* matrixC,const float* matrixA,const float *matrixB,int colsA,int rowsB) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = colsA*(by*BLOCK_SIZE);//A(0,by)
    int aEnd = aBegin + colsA - 1;
    int aStep = BLOCK_SIZE;//offsetA

    int bBegin = BLOCK_SIZE*bx;//B(bx,0)
    int bStep = BLOCK_SIZE*rowsB;//offsetB
    
    float cSub = 0;
    for (int a = aBegin,b = bBegin; a <= aEnd; a += aStep,b += bStep) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        As[ty][tx] = matrixA[a + colsA*ty + tx];
        Bs[ty][tx] = matrixB[b + rowsB*ty + tx];

        __syncthreads();
        
        //i * j for each thread
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            cSub += As[ty][k]*Bs[k][tx];
        }
        __syncthreads();
    }

    int cIndex = (by*BLOCK_SIZE + ty)*rowsB + (bx*BLOCK_SIZE + tx);
    matrixC[cIndex] = cSub;
}


void copyFromCPUToGPU(const float *matrixA, float *d_a, int n) {
    cudaMemcpy(d_a, matrixA, sizeof(float) * n, cudaMemcpyHostToDevice);
}

void copyFromGPUToCPU(const float *d_c, float *matrixC, int n) {
    cudaMemcpy(matrixC, d_c, sizeof(float) * n, cudaMemcpyDeviceToHost);
}

void matrixMulGPU(float* matrixC,const float *matrixA,const float *matrixB,int colsA,int rowsA,int rowsB) {

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, sizeof(float) * colsA*rowsA);   
    cudaMalloc((void**) &d_b, sizeof(float) * rowsB*colsA);  
    cudaMalloc((void**) &d_c, sizeof(float) * rowsB*rowsA); 

    copyFromCPUToGPU(matrixA,d_a,colsA*rowsA);
    copyFromCPUToGPU(matrixB,d_b,rowsB*colsA);

    dim3 blocks(rowsB/BLOCK_SIZE, rowsA/BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    matrixMulGPUKernal0<<<blocks,threads>>>(d_c,d_a,d_b,colsA,rowsA);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf(" - Running time: %f ms\n", time_elapsed);
    double gflop = (2.0 * (double)colsA * colsA * colsA) * 0.000001;
    printf(" - GFlop: %.5f GFlop/sec\n\n", gflop/time_elapsed);

    cudaThreadSynchronize();
    copyFromGPUToCPU(d_c,matrixC,rowsB*rowsA);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void randomInit(float* _data,int size) {
    for (int i = 0; i < size; ++i) {
        _data[i] = rand()/(float)RAND_MAX;
    }
}

bool checkError(const float* matrixA, const float* matrixB, int size) {
    for (int i = 0 ; i < size; ++i) {
        if (fabs(matrixA[i] - matrixB[i]) > 1.0e-3) {
            printf(" ! Wrong index: %d\n", i);
            printf("%f \t %f\n",matrixA[i],matrixB[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {

    if (!InitCUDA()) return 0; 
    srand(63);
    printf("\n - BLOCK_SIZE: %d\n", BLOCK_SIZE);

    int N = (1 << 11);
    int colsA, colsB, colsC, rowsA, rowsB, rowsC;
    colsA = colsB = colsC = rowsA = rowsB = rowsC = N;

    printf(" - Matrix size: %d * %d\n", rowsC, rowsC);

    float* A , *B, *C, *C2;
    A = (float*) malloc(sizeof(float) * colsA * rowsA);
    B = (float*) malloc(sizeof(float) * colsB * rowsB);

    randomInit(A,colsA*rowsA);
    randomInit(B,colsB*rowsB);

    C = (float*) malloc(sizeof(float) * colsC * rowsC);
    memset(C,0,sizeof(float)*colsC*rowsC);
    
    C2 = (float*) malloc(sizeof(float) * colsC * rowsC);
    memset(C2,0,sizeof(float)*colsC*rowsC);
    
    clock_t tick1 = clock();
    matrixMulCPU(C2,A,B,colsA,rowsA,colsB);
    printf(" - CPU use Time : %f ms\n",(double)(clock() - tick1)/CLOCKS_PER_SEC);

    // unsigned int timer = 0;
    // cutilCheckError(cutCreateTimer(&timer));
    // cutilCheckError(cutStartTimer(timer));
    matrixMulGPU(C,A,B,colsA,rowsA,colsB);
    // cutilCheckError(cutStopTimer(timer));
    // printf("GPU use time: %f (ms) \n", cutGetTimerValue(timer));
    // cutilCheckError(cutDeleteTimer(timer));

    if (checkError(C,C2,colsC*rowsC)) {
        printf("Right Answer!\n");
    }else{
        printf("Worng Answer!\n");
    }

    free(A);
    free(B);
    free(C);
    free(C2);

    return 0;
}
