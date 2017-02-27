#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>

#define GET_ARRAY_LEN(array,len) {len = (sizeof(array) / sizeof(array[0]));}

void serialMatrixMultiVec(char **matrix, const char *vec, long long *res, size_t n) {

	for (size_t i = 0; i < n; i++){
		for (size_t j = 0; j < n; j++){
			res[i] += matrix[i][j] * vec[j];
		}
	}
	return ;

}


void optimalMatrixMultiVec(char **matrix, const char *vec, long long *res, size_t n) {

	long long total = 0;

	omp_set_nested(1);

	#pragma omp parallel for //num_threads(4)
	for (size_t i = 0; i < n; i++) {
		#pragma omp parallel for reduction(+:total) //num_threads(4)
		for (size_t j = 0; j < n; j++) {
			total += matrix[i][j] * vec[j];
		}
		res[i] = total;
		total = 0;
	}
	return ;
}

void costOptimalMatrixMultiVec(char **matrix, const char *vec, long long *res, size_t n) {

	long long total = 0;

	#pragma omp parallel for reduction(+:total) //num_threads(4)
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			total += matrix[i][j] * vec[j];
		}
		res[i] += total;
		total = 0;
	}

	return ;
}


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

		long long *res_sel = malloc(n * sizeof(long long));
		long long *res_cost = malloc(n * sizeof(long long));
		long long *res_para = malloc(n * sizeof(long long));
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
		for (size_t i = 0; i < n; i++){
			res_sel[i] = 0;
			res_cost[i] = 0;
			res_para[i] = 0;
		}

		printf("Size N = %d\n", powers[p]);

		double start2 = omp_get_wtime();
		serialMatrixMultiVec(matrix, vec, res_sel, n);
		//printRes(res_sel, 20);
		double end2 = omp_get_wtime();
		printf(" - Serial time: %f \n" , end2 - start2);

		start2 = omp_get_wtime();
		optimalMatrixMultiVec(matrix, vec,res_para, n);
		//printRes(res_para, 20);
		end2 = omp_get_wtime();
		printf(" - Parallel time: %f \n" , end2 - start2);

		start2 = omp_get_wtime();
		costOptimalMatrixMultiVec(matrix, vec, res_cost, n);
		//printRes(res_cost, 20);
		end2 = omp_get_wtime();
		printf(" - Cost opt time: %f \n" , end2 - start2);

		printf("\n");

		for(size_t i = 0; i < n; i++){
			free((void *)matrix[i]);
		}
		free((void *)matrix);

	}

	return 0;

}