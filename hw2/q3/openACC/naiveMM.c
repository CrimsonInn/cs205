# include <stdlib.h>
# include <stdio.h>
# include <math.h>
#include <time.h>
#include <stddef.h>
#include <omp.h>
#include <sys/time.h>

#define GET_ARRAY_LEN(array,len) {len = (sizeof(array) / sizeof(array[0]));}

inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void serialMM(float ** matrix1, float **matrix2, float **res, size_t n){

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                res[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return ;
}


void naiveMM(float ** matrix1, float **matrix2, float **res, size_t n){
    
    #pragma acc parallel loop collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float temp = 0.0;
            for (size_t k = 0; k < n; ++k) {
                temp += matrix1[i][k] * matrix2[k][j];
            }
            res[i][j] = temp;
        }
    }
    return ;
}

int checkError(const float** matrixA, const float** matrixB, size_t n) {
    for (size_t i = 0 ; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (fabs(matrixA[i][j] - matrixB[i][j]) > 1.0e-3) {
                printf(" ! Wrong index: %d %d\n", i, j);
                printf("%f \t %f\n",matrixA[i][j],matrixB[i][j]);
                return 0;
            }
        }
    }
    return 1;
}


int main() {

    int powers[3] = {4, 8, 10};
    int len = 0;
    GET_ARRAY_LEN(powers,len);


    for (int p = 0; p < len; ++p) {

        size_t n = (1 << powers[p]);

        //init input matrix
        float **matrix1 = (float **)malloc(n * sizeof(float *));
        for (size_t i = 0; i < n; ++i) {
            matrix1[i] = (float *)malloc(n * sizeof(float));
        }
        float **matrix2 = (float **)malloc(n * sizeof(float *));
        for (size_t i = 0; i < n; ++i) {
            matrix2[i] = (float *)malloc(n * sizeof(float));
        }
        float **res_sel = (float **)malloc(n * sizeof(float *));
        for (size_t i = 0; i < n; ++i) {
            res_sel[i] = (float *)malloc(n * sizeof(float));
        }
        float **res_para = (float **)malloc(n * sizeof(float *));
        for (size_t i = 0; i < n; ++i) {
            res_para[i] = (float *)malloc(n * sizeof(float));
        }

        #pragma acc parallel loop
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                matrix1[i][j] = 1.0;
                matrix2[i][j] = 2.0;

                res_sel[i][j] = 0;
                res_para[i][j] = 0;
            }
        }

        float gflop = (2.0*n*n*n) * 1E-9;

        double startTime, endTime;

        printf("Matrix size: %d\n", powers[p]);

                 
        //float start2 = omp_get_wtime();
        startTime = seconds();
        serialMM(matrix1, matrix2, res_sel, n);
        //printMatrix(res_sel, 4);
        endTime = seconds() - startTime;
        //float end2 = omp_get_wtime();           
        printf(" - serial gflop: %f \n" , gflop/endTime);
        printf(" - serial time: %f\n", endTime );
        //printf(" - serial time: %f\n", end2 - start2);

        //start2 = omp_get_wtime();
        startTime = seconds();
        naiveMM(matrix1, matrix2, res_para, n);
        endTime = seconds() - startTime;
        //printMatrix(res_para, 4);
        //end2 = omp_get_wtime();
        printf(" - naive openAcc gflop: %f \n" , gflop/endTime);
        printf(" - naive openAcc time: %f\n", endTime);
        //printf(" - naive time: %f\n", end2 - start2);


        if (checkError(res_sel, res_para, n)) {
            printf("Right Answer!\n");
        }else{
            printf("Worng Answer!\n");
        }

        free(matrix1);
        free(matrix2);
        free(res_sel);
        free(res_para);

        printf("\n");

    }

    return 0;

}