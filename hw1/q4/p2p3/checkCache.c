#include "checkCache.h"


int calCache(){

	size_t *sizes_KB = malloc(18 * sizeof(size_t));

	for (int i=0; i<18;i++) sizes_KB[i]=1<<(i+1);
	//random_device rd;
	//mt19937 gen(rd());
	double maxdiff =0.0;
	double previous =0.0;
	size_t chosen_size = 0; 
	printf("calculate cache size...\n");

	for (int i = 0; i < 12; i++)
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
		if (previous!=0.0 && elapsed_secs-previous>maxdiff) {
			maxdiff = elapsed_secs-previous;
			chosen_size = KB(sizes_KB[i-1]);
		}
		previous = elapsed_secs;
		//printf("%d doubles, %f secs, dummy %f \n",sizes,elapsed_secs,dummy);
		/*if (elapsed_secs/2>total/(i+1))
			{
			printf("Found, answer is %d double \n", KB(sizes_KB[i-1]));
			int fit= (int)sqrt(KB(sizes_KB[i-1])/3);
			printf("Good size is below %d \n", fit);
			return fit;
			}*/
	}
	printf("Found, answer is %d MB \n", (chosen_size) >> 17);
	int fit= (int)sqrt(chosen_size/3);
	printf("Good size is below %d \n", fit);
	return fit;
}



