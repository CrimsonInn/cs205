#include <stdio.h>
#include <time.h>

struct csrMat {
    size_t *A;
    size_t *IA;
    size_t *JA;
};

int checkEq(size_t *x, size_t *y, size_t n) {
    for (size_t i=0;i<n;++i) {
        if (x[i] != y[i]) return 0;
    }
    return 1;
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
    printf("total iter num=%lu\n", iter);
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
    size_t *IA = (size_t *) malloc(sizeof(size_t)*(n+1));
    size_t *JA = (size_t *) malloc(sizeof(size_t)*m);
    
    readCSR(IA, JA);
//    csr2edge(IA, JA, n);
    struct csrMat A = {0, IA, JA};
    size_t *x = (size_t *) malloc(sizeof(size_t)*n);
    for (size_t i=0;i<n;++i) x[i] = 0;
    x[0] = 1;
    x[6] = 1;
    
    clock_t start = clock();
    bfs(A, x, n);
    clock_t end = clock();
    printf("elapsed time: %f\n", (double)(end-start)/CLOCKS_PER_SEC);
//  for (size_t i=0;i<n;++i) {
//      printf("%lu,", x[i]);
//  }
//  printf("\n");

    
    return 0;
}
