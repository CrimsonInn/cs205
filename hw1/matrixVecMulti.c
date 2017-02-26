#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>

long long *serial_MatrixMulti(char **matrix, const char *vec, size_t n) {

	long long *res = malloc(n * sizeof(long long));

	for (unsigned long i = 0; i < n; i++){
		res[i] = 0;
	}

	for (unsigned long i = 0; i < n; i++){
		for (unsigned long j = 0; j < n; j++){
			res[i] += matrix[i][j] * vec[j];
		}
	}
	return res;

}


long long  *optimal_MatrixMulti(char **matrix, const char *vec, size_t n) {
	
	long long *res = malloc(n * sizeof(long long));

	for (unsigned long i = 0; i < n; i++){
		res[i] = 0;
	}

	long long total = 0;

	#pragma omp parallel for num_threads(4)
	for (unsigned long i = 0; i < n; i++) {
		#pragma omp parallel for reduction(+:total) num_threads(4)
		for (unsigned long j = 0; j < n; j++) {
			total += matrix[i][j] * vec[j];
			//res[i] += total;
		}
		res[i] = total;
		total = 0;
	}
	return res;
}


long long *cost_optimal_MatrixMulti(char **matrix, const char *vec, size_t n) {

	long long *res = malloc(n * sizeof(long long));

	for (unsigned long i = 0; i < n; i++){
		res[i] = 0;
	}

	long long total = 0;
	//unsigned int i;

	#pragma omp parallel num_threads(4)
	{
		int thread_num = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		long chunk = (long)(n / thread_num);
		
		//#pragma omp parallel for num_threads(4)
		for (unsigned long i  = thread_id * chunk; i < (thread_id + 1) * chunk; i += 1){
			
			#pragma omp for reduction(+:total) 
			for (unsigned long j  = thread_id * chunk; j < (thread_id + 1) * chunk; j += 1) {
				total += matrix[i][j] * vec[j];
			}
			res[i] = total;
			total = 0;

		}
	
	}

	return res;

}


void printRes(long long * nums, size_t n) {
	for (unsigned int i = 0; i < n; ++i) {
		printf("%lld ", nums[i]);
	}
}


int main (int argc, char *argv[]) {

	size_t n = pow(2,16);
	char **matrix = (char **)malloc(n * sizeof(char *));
	for (unsigned int i = 0; i < n; ++i) {
		matrix[i] = (char *)malloc(n * sizeof(char));
	}
	char *vec = malloc(n * sizeof(char));

	//long long *res_sel, *res_para;

	#pragma omp parallel for
	for (unsigned long i = 0; i < n; i++) {
		for(unsigned long j = 0; j < n; j++) {
	 		matrix[i][j] = 1;
		}
	}
	for (unsigned long i = 0; i < n; i++) {
		vec[i] = 1;
	}


	clock_t start = clock();
	long long *res_sel = serial_MatrixMulti(matrix, vec, n);
	printRes(res_sel, 20);
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf(" time %f \n" , seconds);

	start = clock();
	long long *res_para = optimal_MatrixMulti(matrix, vec, n);
	printRes(res_para, 20);
	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf(" time %f \n" , seconds);

	start = clock();
	long long *res_cost = optimal_MatrixMulti(matrix, vec, n);
	printRes(res_cost, 20);
	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf(" time %f" , seconds);



	for(unsigned i = 0; i < n; i++){
		free((void *)matrix[i]);
	}
	free((void *)matrix);

	return 0;

}