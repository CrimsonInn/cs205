#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <math.h>
//double seconds() {
//    struct timeval tp;
//    struct timezone tzp;
//    int i = gettimeofday(&tp, &tzp);
//    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
//}
void mycopy(float * to, float * from, size_t num){
	for (size_t i=0;i<num;i++){
		to[i]=from[i];
	}
}

void apsp_mul(float* dst, float *a, float *b, size_t n) {
	float *tmp = (float *) malloc(sizeof(float)*(n*n));
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


void apsp_add(float* dst, float *a, float *b, size_t n) {
	float *tmp = (float *) malloc(sizeof(float)*(n*n));
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

void APSP(float *A, size_t n) {
	if (n==1) return;
	size_t m = n/2;
	{
	float *a = (float *) malloc(sizeof(float)*(m*m));
	float *b = (float *) malloc(sizeof(float)*(m*m));
	float *c = (float *) malloc(sizeof(float)*(m*m));
	float *d = (float *) malloc(sizeof(float)*(m*m));
	float *temp = (float *) malloc(sizeof(float)*(m*m));

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

void init(float* A, size_t n) {
  float r;
  for (size_t i = 0;i < n; ++i) {
    for (size_t j = 0;j < n; ++j) {
       if (i==j) {
         A[i*n+j]=0.0;
         continue;
       }
       r = (float)rand() / (float)RAND_MAX * 10.0;
       A[i*n+j] = r;
    }
  }
}

int main() {
//	float A[]={0,5,9,-1,-1,0,-1,1,1,-1,0,3,4,-1,5,0};
        //float *A = (float *) malloc(sizeof(float)*8*8);
        size_t psize[]= {1024,1024,256,64};
        for (size_t i = 0;i < 4;++i){
	  size_t n = psize[i];
          printf("Problem Size: %lu\n",n); 
          float *A = (float *) malloc(sizeof(float)*n*n);
          double startTime, endTime;
          init(A, n);
          startTime = clock();
	  APSP(A, n);
          endTime = (double)(clock() - startTime)/CLOCKS_PER_SEC;
	  printf("Openacc %.4fMFLOPs \n", 2*n*n*n/endTime/(1<<20));
          free(A);
        }
        
}
