#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#pragma acc routine
void mycopy(float * to, float * from, size_t num){
	for (size_t i=0;i<num;i++){
		to[i]=from[i];
	}
}

#pragma acc routine
void apsp_mul(float* dst, float *a, float *b, size_t n) {
	float *tmp = (float *) malloc(sizeof(float)*(n*n));
	#pragma acc parallel for collapse(2)
	for (size_t i=0;i < n;i++) {
		for (size_t j=0;j < n;j++) {
			tmp[i*n+j] = -1;
			for (size_t k=0;k < n;k++) {
				if (a[i*n+k]>=0 && b[k*n+j]>=0) {
					if (tmp[i*n+j] < 0 || tmp[i*n+j] > a[i*n+k]+b[k*n+j])
						tmp[i*n+j] = a[i*n+k]+b[k*n+j];
				}
			}
		}
	}
	mycopy(dst, tmp, (n*n));
	free(tmp);
}


#pragma acc routine
void apsp_add(float* dst, float *a, float *b, size_t n) {
	float *tmp = (float *) malloc(sizeof(float)*(n*n));
	#pragma acc parallel for collapse(2)
	for (size_t i=0;i < n;i++) {
		for (size_t j=0;j < n;j++) {
			if (a[i*n+j]>=0 && b[i*n+j]>=0) tmp[i*n+j] = a[i*n+j]<b[i*n+j]?a[i*n+j]:b[i*n+j];
			else if (a[i*n+j]<0 && b[i*n+j]>=0) tmp[i*n+j] = b[i*n+j];
			else if (a[i*n+j]>=0 && b[i*n+j]<0) tmp[i*n+j] = a[i*n+j];
			else if (a[i*n+j]<0 && b[i*n+j]<0) tmp[i*n+j] = -1;
		}
	}
	mycopy(dst, tmp, (n*n));
	free(tmp);
}

#pragma acc routine
void APSP(float *A, size_t n) {
	if (n==1) return;
	size_t m = n/2;
	//#pragma acc data
	{
	float *a = (float *) malloc(sizeof(float)*(m*m));
	float *b = (float *) malloc(sizeof(float)*(m*m));
	float *c = (float *) malloc(sizeof(float)*(m*m));
	float *d = (float *) malloc(sizeof(float)*(m*m));
	float *temp = (float *) malloc(sizeof(float)*(m*m));

	#pragma acc parallel for
	for (size_t i=0;i < n;i++) {
		for (size_t j=0;j < n;j++) {
			if (i < m && j < m) a[i*m + j] = A[i*n + j];
			else if (i < m && j >= m) b[i*m + j-m] = A[i*n + j];
			else if (i >= m && j < m) c[(i-m)*m + j] = A[i*n + j];
			else if (i >= m && j >= m) d[(i-m)*m + j-m] = A[i*n + j];
		}
	}

	APSP(a, m);
	apsp_mul(b, a, b, m);
	apsp_mul(c, c, a, m);

	apsp_mul(temp, c, b, m);
	apsp_add(d, d, temp, m);

	APSP(d, m);
	apsp_mul(b, b, d, m);
	apsp_mul(c, d, c, m);

	apsp_mul(temp, b, c, m);
	apsp_add(a, a, temp, m);

	#pragma acc parallel for
	for (size_t i=0;i < n;i++) {
		for (size_t j=0;j < n;j++) {
			if (i < m && j < m) A[i*n + j] = a[i*m + j];
			else if (i < m && j >= m) A[i*n + j] = b[i*m + j-m];
			else if (i >= m && j < m) A[i*n + j] = c[(i-m)*m + j];
			else if (i >= m && j >= m) A[i*n + j] = d[(i-m)*m + j-m];
		}
	}

	free(a);
	free(b);
	free(c);
	free(d);
	free(temp);
	}
}


int main() {
	float A[]={0,5,9,-1,-1,0,-1,1,1,-1,0,3,4,-1,5,0};
	size_t n = 4;
	//#pragma acc kernels
	{
	APSP(A, n);}
	for (size_t i=0;i < n;i++) {
		for (size_t j=0;j < n;j++) {
			printf("%f,",A[i*n+j]);
		}
		printf("\n");
	}
}