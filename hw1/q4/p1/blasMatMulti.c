#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include <cblas.h>
#include <math.h>

int main()
{
    double *A, *B, *C;
    size_t m, n, k, i, j, l, p, loop;
    size_t LOOP_COUNT;
    double alpha, beta;
    clock_t time_st, time_end;
    float seconds;
    double gflop;
    size_t scale[3] = {1<<6, 1<<10, 1<<16};

    for (p = 0; p < 3; ++p){
      if (p == 0) LOOP_COUNT = 100;
      else LOOP_COUNT = 1;
      m = scale[p];
      k = scale[p];
      n = scale[p];
      alpha = 1.0;
      beta = 0.0;
  
      //A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
      //B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
      //C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
      A = (double *)malloc( m*k*sizeof( double ));
      B = (double *)malloc( k*n*sizeof( double ));
      C = (double *)malloc( m*n*sizeof( double ));
      if (A == NULL || B == NULL || C == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        free(A);
        free(B);
        free(C);
        return 1;
      }

      for (i = 0; i < (m*k); i++) {
          A[i] = (double)(i+1);
      }
  
      for (i = 0; i < (k*n); i++) {
          B[i] = (double)(-i-1);
      }
  
      for (i = 0; i < (m*n); i++) {
          C[i] = 0.0;
      }

      printf("Problem Size n=%lu\n", scale[p]); 
      time_st = clock();
      for (loop = 0; loop < LOOP_COUNT; loop++) {
        for (i = 0; i < m; i++) {
          for (j = 0; j < n; j++) {
            for (l = 0; l < k; l++) {
              C[i*n + j] += A[i*k + l] * B[l*n + j];
            }
          }
        }
      }
      time_end = clock();
      gflop = (2.0*m*n*k)*1E-9;
      seconds = (float)(time_end - time_st) / CLOCKS_PER_SEC;
      printf("Naive Version: %.5f GFlop/sec\n", gflop/seconds/LOOP_COUNT); 
  
      time_st = clock();
      for (loop = 0; loop < LOOP_COUNT; loop++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);
      }
      time_end = clock();
      gflop = (2.0*m*n*k)*1E-9;
      seconds = 1.0*(time_end - time_st) / CLOCKS_PER_SEC;
      printf("BLAS Version: %.5f GFlop/sec\n", gflop/seconds/LOOP_COUNT); 
  
      free(A);
      free(B);
      free(C);
    }
    return 0;
}
