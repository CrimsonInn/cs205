#include "checkCache.h"


int calCache(){

	size_t *sizes_KB = malloc(18 * sizeof(size_t));

	for (int i=0; i<18;i++) sizes_KB[i]=1<<(i+1);
	//random_device rd;
	//mt19937 gen(rd());
	double total =0.0;
	for (int i =0; i<18;i++)
	{
		//uniform_int_distribution<> dis(0,KB(size)-1);
		size_t size = sizes_KB[i];
		int sizes = KB(size);

		double *memory=malloc(sizes* sizeof(double));
		for (int j=0;j<sizes;j++) memory[j]=1;

		double dummy = 0;

		clock_t begin = clock();
		for(int i =0; i< (1<<25);i++) dummy +=memory[rand() % sizes];
		clock_t end = clock();

		double elapsed_secs = (double)(end-begin) / CLOCKS_PER_SEC;
		total+=elapsed_secs;
		//printf("%d doubles, %f secs, dummy %f \n",sizes,elapsed_secs,dummy);
		if (elapsed_secs/2>total/(i+1))
			{
			printf("Found, answer is %d double \n", KB(sizes_KB[i-1]));
			int fit= (int)sqrt(KB(sizes_KB[i-1])/3);
			printf("Good size is below %d \n", fit);
			return fit;
			}
	}
	return 0;
}

