#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>

#define GET_ARRAY_LEN(array,len) {len = (sizeof(array) / sizeof(array[0]));}

long long *serialMatrixMultiVec(char **matrix, const char *vec, size_t n) {

	long long *res = malloc(n * sizeof(long long));

	for (size_t i = 0; i < n; i++){
		res[i] = 0;
	}

	for (size_t i = 0; i < n; i++){
		for (size_t j = 0; j < n; j++){
			res[i] += matrix[i][j] * vec[j];
		}
	}
	return res;

}


long long  *optimalMatrixMultiVec(char **matrix, const char *vec, size_t n) {
	
	long long *res = malloc(n * sizeof(long long));

	for (size_t i = 0; i < n; i++){
		res[i] = 0;
	}

	long long total = 0;

	#pragma omp parallel for //num_threads(4)
	for (size_t i = 0; i < n; i++) {
		#pragma omp parallel for reduction(+:total) //num_threads(4)
		for (size_t j = 0; j < n; j++) {
			total += matrix[i][j] * vec[j];
		}
		res[i] = total;
		total = 0;
	}
	return res;
}

long long  *costOptimalMatrixMultiVec(char **matrix, const char *vec, size_t n) {
	
	long long *res = malloc(n * sizeof(long long));

	for (size_t i = 0; i < n; i++){
		res[i] = 0;
	}

	long long total = 0;

	#pragma omp parallel for reduction(+:total) //num_threads(4)
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			total += matrix[i][j] * vec[j];
		}
		res[i] += total;
		total = 0;
	}

	return res;
}

/*
 * still have bugs in dividing m * m chunk
 */
/*
long long *cost_optimal_MatrixMulti2(char **matrix, const char *vec, size_t n) {

	long long *res = malloc(n * sizeof(long long));

	for (size_t i = 0; i < n; ++i){
		res[i] = 0;
	}

	long long total = 0;
	//unsigned int i;

	#pragma omp parallel
	{
		int thread_num = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		long chunk = (long)(n / thread_num);
		
		#pragma omp for collapse(2)
		for (size_t i  = thread_id * chunk; i < (thread_id + 1) * chunk; ++i){
			
			//#pragma omp parallel for //reduction(+:total) 
			for (size_t j  = thread_id * chunk; j < (thread_id + 1) * chunk; ++j) {
				res[i] += matrix[i][j] * vec[j];
				// += total;
			}
			
			//total = 0;

		}
	
	}

	return res;

}
*/

void printRes(long long * nums, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		printf("%lld ", nums[i]);
	}
	printf("\n");
}


int main (int argc, char *argv[]) {

	//length 
	int powers[3] = {6, 10, 16};
	int len = 0;
	GET_ARRAY_LEN(powers,len);

	for (int p = 0; p < len; ++p){
		
		size_t n = (1 << powers[p]);

		char **matrix = (char **)malloc(n * sizeof(char *));
		for (size_t i = 0; i < n; ++i) {
			matrix[i] = (char *)malloc(n * sizeof(char));
		}
		char *vec = malloc(n * sizeof(char));

		//long long *res_sel, *res_para;

		#pragma omp parallel for
		for (size_t i = 0; i < n; i++) {
			for(size_t j = 0; j < n; j++) {
		 		matrix[i][j] = 1;
			}
		}
		for (size_t i = 0; i < n; i++) {
			vec[i] = 1;
		}

		printf("Size N = %d\n", powers[p]);

		clock_t start = clock();
		long long *res_sel = serialMatrixMultiVec(matrix, vec, n);
		//printRes(res_sel, 20);
		clock_t end = clock();
		float seconds = (float)(end - start) / CLOCKS_PER_SEC;
		printf(" - Serial time: %f \n" , seconds);

		start = clock();
		long long *res_para = optimalMatrixMultiVec(matrix, vec, n);
		//printRes(res_para, 20);
		end = clock();
		seconds = (float)(end - start) / CLOCKS_PER_SEC;
		printf(" - Parallel time: %f \n" , seconds);

		start = clock();
		long long *res_cost = costOptimalMatrixMultiVec(matrix, vec, n);
		//printRes(res_cost, 20);
		end = clock();
		seconds = (float)(end - start) / CLOCKS_PER_SEC;
		printf(" - Cost opt time: %f \n" , seconds);

		printf("\n");

		for(size_t i = 0; i < n; i++){
			free((void *)matrix[i]);
		}
		free((void *)matrix);

	}

	return 0;

}