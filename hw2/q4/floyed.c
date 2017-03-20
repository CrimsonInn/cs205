#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
void mySwap(int *A, int *B, int cA){
	int *C=A;
	A=B;
	B=C;

	printf("A:\n");
    for (int i=0;i<cA*cA;i++){
    	printf("%d ",A[i]);
	}
	printf("\n");
	printf("B:\n");
    for (int i=0;i<cA*cA;i++){
    	printf("%d ",B[i]);
	}
}

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

 const int inf=INFINITY;


void FloydWarshallCPU(float *A,float *result, int cA){
	float *tmp;
	tmp = (float*) malloc(sizeof(float) * cA * cA);
	for(int i=0;i<cA;i++){
		for (int j=0;j<cA;j++){
			tmp[i*cA+j]=A[i*cA+j];
		}
	}

	for(int k=0;k<cA;k++){
		for(int i=0;i<cA;i++){
			for(int j=0;j<cA;j++){
				result[i*cA+j]=min(tmp[i*cA+j],tmp[i*cA+k]+tmp[k*cA+j]);
			}
		}
		if (k<cA-1){
			float *placeholder = tmp;
			tmp= result;
			result= placeholder;
		}
	}
	//if ca%2==0, then result is now pointing to tmp
	if(cA%2==0){
		for(int i=0;i<cA;i++){
		for (int j=0;j<cA;j++){
			tmp[i*cA+j]=result[i*cA+j];
		}
	}
	}
}


void FloydWarshallGPU(float *A,float *result, int cA){

	#pragma acc data copyin(A[0:cA*cA-1]) copyout(result[0:cA*cA-1])
	{
		for(int k=0;k<cA;k++){
			#pragma acc parallel loop collapse(2)
			for(int i=0;i<cA;i++){
				for(int j=0;j<cA;j++){
					result[i*cA+j]=min(A[i*cA+j],A[i*cA+k]+A[k*cA+j]);
				}
			}
			#pragma acc parallel
			if (k<cA-1){
				float *placeholder = A;
				A= result;
				result= placeholder;
			}
		}
	}
}



int main(){
	int cA=(1<<2);
	float* A , *B;
    A = (float*) malloc(sizeof(float) * cA * cA);
    B = (float*) malloc(sizeof(float) * cA * cA);
    
 	static const float coordinates_defaults[16] = 
 		{0,inf,3,inf,
 		2,0,inf,inf,
 		inf,7,0,1,
 		6,inf,inf,0};
 
 	memcpy(A, coordinates_defaults, sizeof(coordinates_defaults));
    //for (int i=0;i<cA*cA;i++){
    //	A[i]=i;
    //	B[i]=cA*cA-1-i;
    //}
    
    //randomInit(B,cA*cA);
    FloydWarshallCPU(A,B,cA);
    printf("A:\n");
    for (int i=0;i<cA*cA;i++){
    	printf("%.0f ",A[i]);
	}
	printf("\n");
	printf("B:\n");
    for (int i=0;i<cA*cA;i++){
    	printf("%.0f ",B[i]);
	}
	printf("GPU version:\n");
    FloydWarshallGPU(A,B,cA);
    printf("A:\n");
    for (int i=0;i<cA*cA;i++){
    	printf("%.0f ",A[i]);
	}
	printf("\n");
	printf("B:\n");
    for (int i=0;i<cA*cA;i++){
    	printf("%.0f ",B[i]);
	}



}