#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>

void printMatrix(double ** matrix, int n) {
	
	for(size_t i = 0; i < n; i++)
		printf("%f ", matrix[i][i]);

	printf("\n");
}

double ** serial_MM(double ** matrix1, double **matrix2, size_t n){

	double **res = (double **)malloc(n * sizeof(double *));
	for (size_t i = 0; i < n; ++i) {
		res[i] = (double *)malloc(n * sizeof(double));
	}
	
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			res[i][j] = 0;
		}
	}

	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			for (size_t k = 0; k < n; ++k) {
				res[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}

	return res;

}

double ** naive_para_MM(double ** matrix1, double **matrix2, size_t n){

	double **res = (double **)malloc(n * sizeof(double *));
	for (size_t i = 0; i < n; ++i) {
		res[i] = (double *)malloc(n * sizeof(double));
	}
	
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			res[i][j] = 0;
		}
	}

	size_t i, j, k;

	#pragma omp parallel shared(matrix1, matrix2, res) private(i, j, k) num_threads(4)
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

/*
	double total = 0.0;

	#pragma omp parallel num_threads(4)
	{
		int thread_num = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		long chunk = (long)(n / thread_num);
		
		//#pragma omp parallel for num_threads(4)
		for (size_t i  = thread_id * chunk; i < (thread_id + 1) * chunk; ++i) {
			for (size_t j  = thread_id * chunk; j < (thread_id + 1) * chunk; ++j) {
				for (size_t k  = thread_id * chunk; k < (thread_id + 1) * chunk; ++k) {



				}
			}
		}




			#pragma omp for reduction(+:total) 
			for (unsigned long j  = thread_id * chunk; j < (thread_id + 1) * chunk; ++j) {
				total += matrix[i][j] * vec[j];
			}

			res[i] = total;
			total = 0;

		}
	
	}
*/
	return res;

}


double ** para_MM(double ** matrix1, double **matrix2, size_t n, int block_size){

	double **res = (double **)malloc(n * sizeof(double *));
	for (size_t i = 0; i < n; ++i) {
		res[i] = (double *)malloc(n * sizeof(double));
	}
	
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			res[i][j] = 0;
		}
	}

	block_size  = 4;

	for (size_t i = 0; i < n; i += block_size) {

		for (size_t j = 0; j < n; j += block_size) {

			#pragma omp parallel for collapse(2)
			for (size_t x = 0; x < block_size; ++x) {

				for (size_t y = 0; y < block_size; ++y) {

					for (size_t k = 0; k < n; ++k) {

						#pragma omp critical
						res[i + x][j + y] += matrix1[i + x][k] * matrix2[k][j + y];
					
					}
				}
			}
		}
	}

	return res;
}


int main (int argc, char *argv[]) {

	size_t n = pow(2,10);

	//init input matrix
	double **matrix1 = (double **)malloc(n * sizeof(double *));
	for (size_t i = 0; i < n; ++i) {
		matrix1[i] = (double *)malloc(n * sizeof(double));
	}

	double **matrix2 = (double **)malloc(n * sizeof(double *));
	for (size_t i = 0; i < n; ++i) {
		matrix2[i] = (double *)malloc(n * sizeof(double));
	}

	#pragma omp parallel for
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			matrix1[i][j] = 1;
			matrix2[i][j] = 2;
		}
	}

	clock_t start = clock();
	double ** res_sel = serial_MM(matrix1, matrix2, n);
	printMatrix(res_sel, 20);
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("serial time %f \n" , seconds);

	start = clock();
	double ** res_para_naive = naive_para_MM(matrix1, matrix2, n);
	printMatrix(res_para_naive, 20);
	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("naive parallel time %f \n" , seconds);

	start = clock();
	double ** res_para = para_MM(matrix1, matrix2, n, 4);
	printMatrix(res_para, 20);
	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("parallel time %f \n" , seconds);

	return 0;
}