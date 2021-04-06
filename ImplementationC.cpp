#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>


#define ARRAY_SIZE 10000
#define NUM_OF_THREADS 8

int array[ARRAY_SIZE];


void change(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}


void quick_sort(int array[], int start, int end)
{
	if (start < end)
	{
		int t = array[end];
		int i = (start - 1);
		for (int j = start; j <= end-1; j++)
		{
			if (array[j] <= t)
			{
				i++;
				change(&array[i], &array[j]);
			}
		}
		change(&array[i + 1], &array[end]);

		int pi = i+1;


		#pragma omp task firstprivate(array, start, pi)
		{
			quick_sort(array, start, pi - 1);
			quick_sort(array, pi + 1, end);
		}
	}
}


void print_array(int a[], int size)
{
		for (int i=0; i < size; i++){ printf("%d ", a[i]); }
		printf("\n");
}


//a basic integer hash
int int_rand(int n)
{
	n = (n << 13U) ^ n;
  n = n * (n * n * 15731U + 789221U) + 1376312589U;
  return n & 0x7fffffffU;
}


int main()
{
	//number generator
	for( int i = 0; i < ARRAY_SIZE-1; i++ )
  {
		array[i] = int_rand(i);
		//array[i] = rand() % 100 + 1; //sometimes faster, sometimes slower
  }

	//To print the Unsorted array
	// printf("\nUnsorted array: \n");
	// print_array(array, ARRAY_SIZE);


	omp_set_num_threads(NUM_OF_THREADS);


	double time = omp_get_wtime();
	#pragma omp parallel
	{
		quick_sort(array, 0, ARRAY_SIZE-1);
	}


	//To print the sorted array
	// printf("\nSorted array: \n");
	// print_array(array, ARRAY_SIZE);


	time = omp_get_wtime() - time;
	std::cout << "\nPart C's time taken: " << time << " seconds" << std::endl;



	return 0;
}
