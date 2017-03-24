#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>
#include <sys/time.h>

/* define computation parameters */
#define TILE_WIDTH 64 /* tile width for matrix multiplication */
#define MAT_SIZE 1024 /* matrix size in 1 dimension assuming square */
#define GANG_SIZE 16 /* gang size */
#define VEC_SIZE 128 /* vector size */

void tile_matmul_acc(const float * mat_a, const float * mat_b, float * mat_c, int mat_size){

    int Idx_m, Idx_n, Idx_k;

/* #pragma acc data copyin(mat_a[:mat_size * mat_size], mat_b[:mat_size * mat_size]) copyout(mat_c[:mat_size * mat_size]) */
/* #pragma acc parallel loop private(Idx_m,Idx_n) tile(TILE_WIDTH,TILE_WIDTH) */
#pragma acc parallel loop tile(TILE_WIDTH,TILE_WIDTH)
/* #pragma acc parallel num_gangs(GANG_SIZE) vector_length(VEC_SIZE) */
/* #pragma acc loop independent gang */
    for (Idx_m = 0; Idx_m < mat_size; Idx_m++){
            for (Idx_n = 0; Idx_n < mat_size; Idx_n++){
                float temp_sum = 0;
		/* #pragma acc loop independent vector reduction(+:temp_sum) */
                for (Idx_k = 0; Idx_k < mat_size; Idx_k++){
                    temp_sum += mat_a[Idx_m * mat_size + Idx_k] * mat_b[Idx_k * mat_size + Idx_n];
                }
                mat_c[Idx_m * mat_size + Idx_n] = temp_sum;
            }
    } 
}

inline double time_now() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/* function to randomly assign values for matrix */
void randmat(float * matrix, int nsize){
    for(int idx=0; idx < nsize * nsize; idx++){
    matrix[idx] = (double)(rand())/ ( (double)(RAND_MAX) + 1.0);
    }
}

/* main function */
int main(int argc, char *argv[]){

    /* print matrix size n_now */
    const int n_now = MAT_SIZE;
    printf(" --- Matrix multiplicatoin --- \n");
    printf("Matrix size n is %d\n", n_now);
    printf("Gang size is %d\n", GANG_SIZE);
    printf("Vector size is %d\n", VEC_SIZE);
    
    /* define matrix in host and in device */
    float *host_a, *host_b, *c_tile_acc;
    
    /* size of matrix in byte */
    size_t mat_in_byte = (size_t)n_now * (size_t)n_now * sizeof(float);
    
    /* randomly setup host_a and host_b */
    host_a = (float *) malloc(mat_in_byte);
    host_b = (float *) malloc(mat_in_byte);
    randmat(host_a, n_now);
    randmat(host_b, n_now);

    /* set c_tile_acc c_naive to zeros */
    c_tile_acc = (float *) malloc(mat_in_byte);
    memset(c_tile_acc, 0, mat_in_byte);
    /* memset(c_naive, 0, mat_in_byte); */

    /* call tile_matmul_acc */
    double start_time, elapsed_time;
    /* printf("Start tile_matmul_acc\n"); */
    start_time = time_now();
    tile_matmul_acc(host_a, host_b, c_tile_acc, n_now);
    elapsed_time = time_now() - start_time;
    /* printf("End tile_matmul_acc\n"); */
    printf("tile_matmul_acc elapsed time is %f sec\n", elapsed_time);
    printf("Throughput is %f GFlop/s\n", 2.0 * (double)n_now * (double)n_now * (double)n_now / ((double)elapsed_time) * 1.e-9);
    
    /* clean up memory */
    free(host_a);
    free(host_b);
    free(c_tile_acc);
    
    return 0;
}
