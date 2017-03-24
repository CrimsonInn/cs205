/*
    CS205 HW2 Part 3.1
    The following code was written with reference to nVidia tutorial
    and Programming Massively Parallel Processors.
*/

#include <stdio.h>
#include <cublas_v2.h>

/* define computation parameters */
#define TILE_WIDTH 32 /* tile width for matrix multiplication */
#define MAT_SIZE 1024 /* matrix size in 1 dimension assuming square */
#define BLOCK_SIZE_X 32 /* number of threads in a block in x direction */
#define BLOCK_SIZE_Y 32 /* number of threads in a block in y direction */

/* GPU kernel: tile matrix multiplcation */
__global__ void tile_matmul(float const * const mat_a, float const * const mat_b, float * const mat_c, const int mat_size){

    /* save indices into registers */
    int tIdx_x = threadIdx.x; /* thread index in x */
    int tIdx_y = threadIdx.y; /* thread index in y */
    int bIdx_x = blockIdx.x; /* block index in x */
    int bIdx_y = blockIdx.y; /*block index in y */
    
    /* row and column of mat_c to calculate now */
    int idx_row = bIdx_y * TILE_WIDTH + tIdx_y;
    int idx_col = bIdx_x * TILE_WIDTH + tIdx_x;
    
    /* store tiles of input mat_a and mat_b in shared memory */
    __shared__ float shared_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_b[TILE_WIDTH][TILE_WIDTH];
    
    /* store temp sum */
    float temp_sum = 0;
    
    /* loop through all tile phases */
    for (int tile_phase = 0; tile_phase < mat_size/TILE_WIDTH; tile_phase++){
        
        /* load a tile of mat_a and mat_b from global memory to shared memory */
        shared_a[tIdx_y][tIdx_x] = mat_a[idx_row * mat_size + tile_phase * TILE_WIDTH + tIdx_x];
        shared_b[tIdx_y][tIdx_x] = mat_b[(tile_phase * TILE_WIDTH + tIdx_y) * mat_size + idx_col];
        /* synchronize threads in a tile */
        __syncthreads();
        
        for (int idx_el = 0; idx_el < TILE_WIDTH; idx_el++){
            temp_sum += shared_a[tIdx_y][idx_el] * shared_b[idx_el][tIdx_x];
        }
        __syncthreads();
    }
    mat_c[idx_row * mat_size + idx_col] = temp_sum;
}


/* function to randomly assign values for matrix */
void randmat(float * matrix, int nsize){
    for(int idx=0; idx < nsize * nsize; idx++){
    matrix[idx] = double(rand())/ (double(RAND_MAX) + 1.0);
    }
}


/* function to calculate max error between mat_1 and mat_2 */
void max_err(float * mat_1, float * mat_2, int nsize){
    float err = 0.0;
    for(int idx=0; idx < nsize * nsize; idx++){
        err = max(err, abs(( (float)mat_1[idx] - (float)mat_2[idx]) / (float)mat_1[idx] ) );
    }
    printf("Max error is %e percent \n", err*100.0);
}

/* main function */
int main(int argc, char *argv[]){

    /* check GPU info on Odyssey */
    int dev;
    cudaDeviceProp prop;

    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    printf(" --- General info --- \n");
    printf("GPU name: %d %s\n", dev, prop.name);
    printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
    printf("Clock rate:  %d\n", prop.clockRate);

    printf(" --- Memory info --- \n");
    printf("Total global mem:  %ld\n", prop.totalGlobalMem);

    printf(" --- Multiprocessor (MP) Info --- \n");
    printf("MP count: %d\n", prop.multiProcessorCount);
    printf("Shared mem per MP:  %ld\n", prop.sharedMemPerBlock);
    printf("Registers per MP:  %d\n", prop.regsPerBlock);
    printf("Threads in warp:  %d\n", prop.warpSize);
    printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
    printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0],
        prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
        prop.maxGridSize[1], prop.maxGridSize[2]);
    printf( "\n" );

    /* print matrix size n_now */
    const int n_now = MAT_SIZE;
    printf(" --- Matrix multiplicatoin --- \n");
    printf("Matrix size n is %d\n", n_now);

    /* define matrix in host and in device */
    float *host_a, *host_b, *c_cublas, *c_naive;
    float *dev_a, *dev_b, *dev_c;

    /* size of matrix in byte */
    size_t mat_in_byte = (size_t)n_now * (size_t)n_now * sizeof(float);

    /* randomly setup host_a and host_b */
    host_a = (float *) malloc(mat_in_byte);
    host_b = (float *) malloc(mat_in_byte);
    randmat(host_a, n_now);
    randmat(host_b, n_now);

    /* set c_cublas and c_naive to zeros */
    c_cublas = (float *) malloc(mat_in_byte);
    c_naive = (float *) malloc(mat_in_byte);
    memset(c_cublas, 0, mat_in_byte);
    memset(c_naive, 0, mat_in_byte);

    /* setup dev_a, dev_b, and dev_c */
    cudaMalloc((void **)&dev_a, mat_in_byte);
    cudaMalloc((void **)&dev_b, mat_in_byte);
    cudaMalloc((void **)&dev_c, mat_in_byte);
    cudaMemcpy(dev_a, host_a, mat_in_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, mat_in_byte, cudaMemcpyHostToDevice);
    cudaMemset(dev_c, 0, mat_in_byte);

    /* setup CUDA timer */
    float time_tile, time_cublas;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*
        tile_matmul kernel computation
    */

    /* preparation */
    dim3 dim_thread(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 dim_block( ceil(n_now / BLOCK_SIZE_X), ceil(n_now / BLOCK_SIZE_Y), 1);

    /* start timer */
    cudaEventRecord(start, 0);

    /* core computation */
    tile_matmul<<<dim_block, dim_thread>>> (dev_a, dev_b, dev_c, n_now);
    
    /* end timer */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tile, start, stop);

    /* wrap up */
    cudaMemcpy(c_naive, dev_c, mat_in_byte, cudaMemcpyDeviceToHost);
    cudaMemset(dev_c, 0, mat_in_byte);
    printf("tile_matmul elapsed time is %f seconds\n", time_tile / 1000.0f);
    printf("Throughput is %f GFlop/s\n", 2.0 * (double)n_now * (double)n_now * (double)n_now /
    ((double)time_tile * 1.e-3) * 1.e-9);


    /*
        CUBLAS computation
    */

    /* preparation */
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;

    /* start timer */
    cudaEventRecord(start, 0);

    /* core computation */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_now, n_now, n_now,
        (float *)&alpha, (float *)dev_b, n_now, (float *)dev_a, n_now,
        (float *)&beta, (float *)dev_c, n_now);
    cudaThreadSynchronize();

    /* end timer */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cublas, start, stop);

    /* wrap up */
    cudaMemcpy(c_cublas, dev_c, mat_in_byte, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    printf("CUBLAS elapsed time is %f seconds\n", time_cublas / 1000.0f);
    printf("Throughput is %f GFlop/s\n", 2.0 * (double)n_now * (double)n_now * (double)n_now /
    ((double)time_cublas * 1.e-3) * 1.e-9);

    /* compare computation results */
    max_err(c_cublas, c_naive, n_now);

    /* clean up memory */
    free(host_a);
    free(host_b);
    free(c_cublas);
    free(c_naive);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaDeviceReset();

    return 0;
}
