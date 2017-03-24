#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stddef.h>
#include <sys/time.h>
//#include <omp.h>

double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
struct csrMat {
    size_t *A;
    size_t *IA;
    size_t *JA;
};

#pragma acc routine
int checkEq(size_t *x, size_t *y, size_t n) {
    for (size_t i=0;i<n;++i) {
        if (x[i] != y[i]) return 0;
    }
    return 1;
}
#pragma acc routine
void mycopy(float * to, float * from, size_t num){
    for (size_t i=0;i<num;i++){
        to[i]=from[i];
    }
}
void bfs_new(struct csrMat A, size_t *x, size_t n, size_t m){
    size_t *xp = (size_t *) malloc(sizeof(size_t)*n);
    memcpy(xp, x, sizeof(size_t)*n);
    size_t iter = 0;
    size_t *AA=A.A;
    size_t *IA=A.IA;
    size_t *JA=A.JA;
    //printf("\n begin \n");
    #pragma acc data copyin(xp[0:n],IA[0:n+1],JA[0:m]) copyout(x[0:n])
    {
    while (iter<2000) {
        iter++;
        //printf("\n iter %zu \n",iter);
        #pragma acc parallel loop
        for(int i=0;i<n;i++){
            double sum=0;
            int row_start=IA[i];
            int row_end=IA[i+1];
            #pragma acc loop reduction(+:sum)
            for(int j=row_start;j<row_end;j++){
                unsigned int Acol=JA[j];
                size_t Acoef = 1;
                size_t xcoef = xp[Acol];
                sum+=Acoef & xcoef;
            }

            x[i]=sum>0 ? 1:0;
        }
        //printf("%zu",iter);
        if (checkEq(x, xp, n)) break;
        else {
            memcpy(xp, x, sizeof(size_t)*n);
        }
    }
    }
    free(xp);
    //printf("total iter num=%lu\n", iter);
    return;
}



void bfs_fast(struct csrMat A, size_t *x, size_t n,size_t m) {
    size_t *xp = (size_t *) malloc(sizeof(size_t)*n);
    memcpy(xp, x, sizeof(size_t)*n);
    size_t iter = 0;
    size_t *AA=A.A;
    size_t *IA=A.IA;
    size_t *JA=A.JA;
    //printf("\n begin \n");
    #pragma acc data copyin(xp[0:n],IA[0:n+1],JA[0:m]) copyout(x[0:n])
    {
    while (1) {
        iter++;
        #pragma acc kernels
        for (size_t i = 0;i < n; ++i) {
            //#pragma acc parallel loop
            for (size_t j=(IA)[i]; j < (IA)[i+1]; ++j) {
                if (xp[JA[j]] > 0) {
                    if (xp[i] == 0 || xp[JA[j]]+1 < x[i])
                        x[i] = xp[JA[j]]+1;
                }
            }
        }
        if (checkEq(x, xp, n)) break;
        else {
            memcpy(xp, x, n);
        }
    }
    }
    //printf("total iter num=%lu\n", iter);
    return;
}

void bfs_fast2(struct csrMat A, size_t *x, size_t n,size_t m) {
    size_t *xp = (size_t *) malloc(sizeof(size_t)*n);
    memcpy(xp, x, sizeof(size_t)*n);
    size_t iter = 0;
    size_t *AA=A.A;
    size_t *IA=A.IA;
    size_t *JA=A.JA;
    //printf("\n begin \n");
    #pragma acc data copyin(xp[0:n],IA[0:n+1],JA[0:m]) copyout(x[0:n])
    {
    while (1) {
        iter++;
        #pragma acc parallel loop
        for (size_t i = 0;i < n; ++i) {
            //#pragma acc parallel loop
            for (size_t j=(IA)[i]; j < (IA)[i+1]; ++j) {
                if (xp[JA[j]] > 0) {
                    if (xp[i] == 0 || xp[JA[j]]+1 < x[i])
                        x[i] = xp[JA[j]]+1;
                }
            }
        }
        if (checkEq(x, xp, n)) break;
        else {
            memcpy(xp, x, sizeof(size_t)*n);
        }
    }
    }
    //printf("total iter num=%lu\n", iter);
    return;
}
void bfs(struct csrMat A, size_t *x, size_t n) {
    size_t *xp = (size_t *) malloc(sizeof(size_t)*n);
    memcpy(xp, x, sizeof(size_t)*n);
    size_t iter = 0;
    while (1) {
        iter++;
        for (size_t i = 0;i < n; ++i) {
            for (size_t j=(A.IA)[i]; j < (A.IA)[i+1]; ++j) {
                if (xp[A.JA[j]] > 0) {
                    if (xp[i] == 0 || xp[A.JA[j]]+1 < x[i])
                        x[i] = xp[A.JA[j]]+1;
                }
            }
        }
        if (checkEq(x, xp, n)) break;
        else {
            memcpy(xp, x, sizeof(size_t)*n);
        }
    }
    //printf("total iter num=%lu\n", iter);
    return;
}



void csr2edge(size_t *IA, size_t *JA, size_t n) {
    size_t curRow = 0;
    while (curRow < n) {
        for (size_t j=IA[curRow]; j < IA[curRow+1]; ++j) {
            printf("%lu\t%lu\n", curRow, JA[j]);
        }
        curRow++;
    }

}

void readCSR(size_t *IA, size_t *JA) {
    FILE * file;
    size_t n1, n2;
    file = fopen("bfsdat.txt" , "r");
    size_t curRow = 0;
    size_t curRowNz = 0;
    IA[0] = 0;
    if (file) {
        while (fscanf(file, "%lu\t%lu", &n1, &n2) != EOF) {
//                        printf( "%lu, %lu\n", n1, n2);
//                        printf( "%lu, %lu\n", n1, curRow);
            
            while (n1 != curRow) {
                curRow++;
                IA[curRow] = IA[curRow-1] + curRowNz;
                curRowNz = 0;
            }
            
            JA[IA[curRow] + curRowNz]=n2;
            curRowNz++;
        }
        fclose(file);
    }
}


int main(int argc, const char * argv[]) {
//    size_t IA[8] = {0,1,2,5,7,9,11,13};
//    size_t JA[13] = {3,0,3,5,6,0,6,1,6,2,4,1,3};
//    csr2edge(IA, JA, 7);
//    struct csrMat A = {0, IA, JA};
//    size_t *x = (size_t *) malloc(sizeof(size_t)*n);
//    for (size_t i=0;i<n;++i) x[i] = 0;
//    x[0] = 1;
//    x[6] = 1;
//    
//    bfs(A, x, n);
//    for (size_t i=0;i<n;++i) {
//        printf("%lu,", x[i]);
//    }
//    printf("\n");

    
    size_t n = 1024;
    size_t m = 20000;
    printf("prepare 1");
    size_t *IA = (size_t *) malloc(sizeof(size_t)*(n+1));
    size_t *JA = (size_t *) malloc(sizeof(size_t)*m);
    
    readCSR(IA, JA);
    printf("prepare 2");
//    csr2edge(IA, JA, n);
    struct csrMat A = {0, IA, JA};
    size_t *x = (size_t *) malloc(sizeof(size_t)*n);
    size_t *xt = (size_t *) malloc(sizeof(size_t)*n);
    for (size_t i=0;i<n;++i) x[i] = 0;
    x[0] = 1;
    x[6] = 1;
    memcpy(xt, x, sizeof(size_t)*n);
    
    //printf("start 1");
    double start = seconds();
    for (size_t z=0;z<10;z++) {
      bfs(A, xt, n);
      memcpy(xt, x, sizeof(size_t)*n);
    }
    double end = seconds();
    printf("Basic version: %fsec\n", (double)(end-start));
    //printf("start 2");
    start = seconds();
    for (size_t z=0;z<10;z++) {
      bfs_new(A, xt, n,m);
      memcpy(xt, x, sizeof(size_t)*n);
    }
    end = seconds();
    printf("OpenAcc version: %fsec\n", (double)(end-start));
        //printf("start 3");
    start = seconds();
    for (size_t z=0;z<10;z++) {
      bfs_fast(A, xt, n,m);
      memcpy(xt, x, sizeof(size_t)*n);
    }
    end = seconds();
    printf("OpenAcc version2: %fsec\n", (double)(end-start));
//  for (size_t i=0;i<n;++i) {
//      printf("%lu,", x[i]);
//  }
//  printf("\n");

    
    return 0;
}
