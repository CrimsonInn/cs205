#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>
#include "checkCache.h"

#define GET_ARRAY_LEN(array,len) {len = (sizeof(array) / sizeof(array[0]));}


void printMatrix(double ** matrix, int n) {
	
	for(size_t i = 0; i < n; i++)
		printf("%f ", matrix[i][i]);

	printf("\n");
}

void serialMM(double ** matrix1, double **matrix2, double **res, size_t n){

	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			for (size_t k = 0; k < n; ++k) {
				res[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}

	return ;

}

void naiveParaMM(double ** matrix1, double **matrix2, double **res, size_t n){
	
	size_t i, j, k;

	#pragma omp parallel shared(matrix1, matrix2, res) private(i, j, k) //num_threads(4)
    {  
        #pragma omp for 
        for(i = 0; i < n; ++i)  
        {  
            for(j = 0; j < n; ++j)  
            {  
                for(k = 0; k < n; ++k)  
                {  
                    res[i][j] += matrix1[i][k] * matrix2[k][j];  
                }  
            }  
        }  
    }

	return ;

}


void paraMM(double ** matrix1, double **matrix2, double **res, size_t n, int block_size){

	//block_size  = 256;

	//#pragma omp parallel for num_threads(4) collapse(2)
	#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < n; i += block_size) {

		for (size_t j = 0; j < n; j += block_size) {


			for (size_t x = 0; x < block_size; ++x) {

				for (size_t y = 0; y < block_size; ++y) {
					double temp=0.0;
					for (size_t k = 0; k < n; ++k) {
						//#pragma omp critical
						temp += matrix1[i + x][k] * matrix2[k][j + y];
					
					}
					res[i + x][j + y]=temp;
				}
			}
			
		}
	}

	return ;
}


int main (int argc, char *argv[]) {

	int powers[3] = {4, 8, 10};
	int len = 0;
	GET_ARRAY_LEN(powers,len);

	int cacheTemp = calCache();
	int num_blocks = 0;

	for(unsigned int i = 31; i >= 0; --i){
		unsigned int flag = (1 << i);
		if(flag & cacheTemp){
			cacheTemp = flag & cacheTemp;
			break;
		}
	}

	for (int p = 0; p < len; ++p){

		size_t n = (1 << powers[p]);

		num_blocks = 256 * 2 >= n ? n / 8 : 256;

		//init input matrix
		double **matrix1 = (double **)malloc(n * sizeof(double *));
		for (size_t i = 0; i < n; ++i) {
			matrix1[i] = (double *)malloc(n * sizeof(double));
		}

		double **matrix2 = (double **)malloc(n * sizeof(double *));
		for (size_t i = 0; i < n; ++i) {
			matrix2[i] = (double *)malloc(n * sizeof(double));
		}

		//init result
		double **res_sel = (double **)malloc(n * sizeof(double *));
		for (size_t i = 0; i < n; ++i) {
			res_sel[i] = (double *)malloc(n * sizeof(double));
		}

		double **res_para_naive = (double **)malloc(n * sizeof(double *));
		for (size_t i = 0; i < n; ++i) {
			res_para_naive[i] = (double *)malloc(n * sizeof(double));
		}

		double **res_para = (double **)malloc(n * sizeof(double *));
		for (size_t i = 0; i < n; ++i) {
			res_para[i] = (double *)malloc(n * sizeof(double));
		}

		#pragma omp parallel for
		for (size_t i = 0; i < n; ++i) {
			for (size_t j = 0; j < n; ++j) {
				matrix1[i][j] = 1;
				matrix2[i][j] = 2;

				res_sel[i][j] = 0;
				res_para[i][j] = 0;
				res_para_naive[i][j] = 0;
			}
		}

		printf("Block size: %d\n", num_blocks);
		printf("Matrix size: %d\n", powers[p]);

		clock_t start = clock();
		serialMM(matrix1, matrix2, res_sel, n);
		//printMatrix(res_sel, 4);
		clock_t end = clock();
		float seconds = (float)(end - start) / CLOCKS_PER_SEC;
		printf(" - serial time %f \n" , seconds);

		start = clock();
		naiveParaMM(matrix1, matrix2, res_para_naive, n);
		//printMatrix(res_para_naive, 4);
		end = clock();
		seconds = (float)(end - start) / CLOCKS_PER_SEC;
		printf(" - naive parallel time %f \n" , seconds);

		start = clock();
		paraMM(matrix1, matrix2, res_para, n, num_blocks);
		//printMatrix(res_para, 4);
		end = clock();
		seconds = (float)(end - start) / CLOCKS_PER_SEC;
		printf(" - parallel time %f \n" , seconds);

		free(matrix1);
		free(matrix2);

		printf("\n");

	}

	return 0;
}
