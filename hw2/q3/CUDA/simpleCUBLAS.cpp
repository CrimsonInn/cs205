#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

/* Main */
int main(int argc, char **argv)
{

    int powers[3] = {6, 8, 10};
    size_t N = (1 << 6);

    cublasStatus_t status;
    float *h_A;
    float *h_B;
    float *h_C;
    float *h_C_ref;
    float *d_A = 0;
    float *d_B = 0;
    float *d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;
    cublasHandle_t handle;

    /* Initialize CUBLAS */
    printf("matrix multiplication CUBLAS test running..\n");

    status = cublasCreate(&handle);

    h_A = (float *)malloc(n2 * sizeof(h_A[0]));

    h_B = (float *)malloc(n2 * sizeof(h_B[0]));

    h_C = (float *)malloc(n2 * sizeof(h_C[0]));


    for (i = 0; i < n2; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);

    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

    status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

    /* Performs operation using plain C code */
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;


    //timer
    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* Performs operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf(" - Running time: %f ms\n", time_elapsed);
    double gflop = (2.0 * (double)N * N * N) * 0.000001;
    printf(" - GFlop: %.5f GFlop/sec\n", gflop/time_elapsed);


    /* Allocate host memory for reading back the result from device memory */
    h_C = (float *)malloc(n2 * sizeof(h_C[0]));


    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);


    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;

    for (i = 0; i < n2; ++i)
    {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    /* Shutdown */
    status = cublasDestroy(handle);

    if (error_norm / ref_norm < 1e-6f)
    {
        printf("matrix multiplication CUBLAS test passed.\n");
        exit(EXIT_SUCCESS);
    }
    else
    {
        printf("matrix multiplication CUBLAS test failed.\n");
        exit(EXIT_FAILURE);
    }
}
