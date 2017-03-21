# include <stdlib.h>
# include <stdio.h>
# include <math.h>
#include <time.h>
#include <stddef.h>
#include <sys/time.h>


inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void serialMM(float * matrixA, float *matrixB, float *res, size_t n){

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                res[i * n + j] += matrixA[i * n + k] * matrixB[k *n + j];
            }
        }
    }
    return ;
}


int checkError(const float* matrixA, const float* matrixB, size_t n) {
    for (size_t i = 0 ; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (fabs(matrixA[i * n + j] - matrixB[i * n + j]) > 1.0e-3) {
                printf(" ! Wrong index: %d %d\n", i, j);
                printf("%f \t %f\n",matrixA[i * n + j],matrixB[i * n + j]);
                return 0;
            }
        }
    }
    return 1;
}


void randomInit(float* _data, int size) {
    for (int i = 0; i < size; ++i) {
        _data[i] = rand()/(float)RAND_MAX;
    }
}


int main() {

    int N = (1 << 10); //pow(2,10); 

    float *A, *B, *C, *D;
    A = (float*) malloc(sizeof(float) * N * N);
    B = (float*) malloc(sizeof(float) * N * N);
    C = (float*) malloc(sizeof(float) * N * N);
    // D = (float*) malloc(sizeof(float) * N * N);
    
    srand(63);
    randomInit(A, N * N);
    randomInit(B, N * N);
    memset(C, 0, sizeof(float) * N * N);


    // serialMM(A, B, D, N);
    double gflop = (2.0 * (double)N * N * N) * 0.000000001;
    double startTime, exeTime;
    startTime = seconds();

    // openAcc tile
    #pragma acc data copyin(A[:N*N],B[:N*N]) copyout(C[:N*N])
    #pragma acc parallel loop independent tile(16, 16)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float temp = 0.0;
            // #pragma acc loop independent reduction(+:temp)
            for (int k = 0; k < N; ++k) {
                temp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = temp;
        }
    }

    // openAcc gang vector with parameters
    // #pragma acc data copyin(A[:N*N],B[:N*N]) copyout(C[:N*N])
    // #pragma acc parallel num_gangs(64) vector_length(256)  
    // #pragma acc loop independent gang collapse(2)
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         float temp = 0.0;
    //         #pragma acc loop independent vector reduction(+:temp)
    //         for (int k = 0; k < N; ++k) {
    //             temp += A[i * N + k] * B[k * N + j];
    //         }
    //         C[i * N + j] = temp;
    //     }
    // }

    // open Acc gang vector with default parameters
    // #pragma acc data copyin(A[:N*N],B[:N*N]) copy(C[:N*N])
    // #pragma acc parallel loop independent gang collapse(2)
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         float temp = 0.0;
    //         #pragma acc loop independent vector reduction(+:temp)
    //         for (int k = 0; k < N; ++k) {
    //             temp += A[i * N + k] * B[k * N + j];
    //         }
    //         C[i * N + j] = temp;
    //     }
    // }

    exeTime = seconds() - startTime;

    printf(" - Time: %f ms \n", exeTime * 1000.0);
    printf(" - GFlop: %.5f GFlop/sec\n\n", gflop/exeTime);

    // if(checkError(D, C, N)){
    //     printf("Right!\n");
    // }else{
    //     printf("Wrong!\n");
    // }

    free(A);
    free(B);
    free(C);
    // free(D);

    return 0;

}