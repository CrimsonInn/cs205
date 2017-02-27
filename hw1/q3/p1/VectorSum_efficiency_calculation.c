/*
 ============================================================================
 Name        : test2.c
 Author      : hengte lin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello OpenMP World in C
 ============================================================================
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>
//#include "VecotrSum.h"
/**
 * Hello OpenMP World prints the number of threads and the current thread id
 */
long long serial_sum(const char* a,size_t n);
long long time_optimal_sum(const char* a,size_t n);
long long cost_optimal_sum(const char* a,size_t n);

int main (int argc, char *argv[]) {

size_t options[]={pow(2,6),pow(2,10),pow(2,20),pow(2,32)};
for(int i=0;i<4;i++){
	printf("size %d \n",options[i]);
	for (int j=2;j<17;j++){
		
		omp_set_num_threads(j);
		printf("thread %d \n",j);
		test_sum(options[i]);
	}
	}
}

int test_sum (size_t n) {

	char *a = malloc(n * sizeof(char));
	if (a == NULL) {
		 perror("malloc");
		 return 1;
	}
	unsigned long i;

#pragma omp parallel for
	for (i=0;i<n;i++){
	 a[i]=1;
	}
	//printf("true threads %d \n", omp_get_max_threads());
	double start = omp_get_wtime();
	serial_sum(a,n);
	double elapsed1 = omp_get_wtime()-start;
	//printf(" time %f \n" , elapsed1);

	start = omp_get_wtime();
	time_optimal_sum(a,n);
	double elapsed2 = omp_get_wtime()-start;
	//printf(" time %f \n" , elapsed2);
	//printf("clock time %f \n" , seconds);
	printf("%f, %f \n",elapsed1/elapsed2,(elapsed1/elapsed2)/omp_get_max_threads());

	start = omp_get_wtime();
	cost_optimal_sum(a,n);
	double elapsed3 = omp_get_wtime()-start;
	//printf(" time %f \n" , elapsed3);
	printf("%f, %f \n",elapsed1/elapsed3,(elapsed1/elapsed3)/omp_get_max_threads());

	free(a);
	return 0;
}


long long serial_sum(const char *a, size_t n){
	long long total=0.;
	size_t i;
	for(i=0;i<n;i++){
		total+=a[i];
	}
	return total;
}

long long time_optimal_sum(const char *a, size_t n){
	long long total=0.;
	size_t i;
#pragma omp parallel for reduction(+:total) //num_threads(4)
		for(i=0;i<n;i++){
			total+=a[i];
		}

	return total;
}

long long cost_optimal_sum(const char *a, size_t n){
	long long total=0.;
	long long i;
	//double logn = log((double) n);
	//int threads = (int) (n/logn);
	//printf("num threads: ", threads);

#pragma omp parallel reduction(+:total) private(i) //num_threads(4)
	{
	int thread_num = omp_get_num_threads();
	int thread_id = omp_get_thread_num();
	long chunk = (long)(n/thread_num);
	for (i=thread_id*chunk;i<(thread_id+1)*chunk;i+=1){
		total+=a[i];
		}

	}
	return total;
}

