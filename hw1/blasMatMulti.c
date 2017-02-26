#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <math.h>

int main()
{
    double *A, *B, *C;
    int m, n, k, i, j, l, p, loop;
    size_t LOOP_COUNT = 1;
    double alpha, beta;
    double time_st, time_end, gflop;

    size_t scale[3] = {pow(2, 6), pow(2,10), pow(2,16)};
    for (p = 0; p < 3; ++p){

      m = scale[p], k = scale[p], n = scale[p];
      alpha = 1.0; beta = 0.0;
  
      A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
      B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
      C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
      if (A == NULL || B == NULL || C == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
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
      time_st = dsecnd();
      for (loop = 0; loop < LOOP_COUNT; loop++) {
        for (i = 0; i < m; i++) {
          for (j = 0; j < n; j++) {
            for (l = 0; l < k; l++) {
              C[i*n + j] += A[i*k + l] * B[l*n + j];
            }
          }
        }
      }
      time_end = dsecnd();
      gflop = (2.0*m*n*k)*1E-9;

      printf("Naive Version: %.5f GFlop/sec\n", gflop/(time_end-time_st)/LOOP_COUNT); 
  
      time_st = dsecnd();
      for (loop = 0; loop < LOOP_COUNT; loop++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);
      }
      time_end = dsecnd();
      gflop = (2.0*m*n*k)*1E-9;
      printf("MKL BLAS Version: %.5f GFlop/sec\n", gflop/(time_end-time_st)/LOOP_COUNT); 
  
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
    }
    return 0;
}
