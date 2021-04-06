// ImplementationA.cpp
// compile: g++ -o main -fopenmp "C:\Users\User1\Documents\openmp\Implementation.cpp"
// run: ./main "C:\Users\User1\Documents\openmp\inputImage.pgm" "C:\Users\User1\Documents\openmp\outputImage.pgm" 25 8 A1

#include <omp.h>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>

int height;
int width;
int inputImage[1000][1000];
int outputImage[1000][1000];
int chunkSize;
int threadsCount;

int maskX[3][3], maskY[3][3];
int *order;



int clamp(int v, int min, int max)
{
	return std::max(std::min(v, max), min);
}


// Part A1
void perwitt_static()
{
	#pragma omp parallel shared(inputImage,outputImage,chunkSize,order)
	{
		#pragma omp for schedule(static,chunkSize) nowait
		for(int x = 0; x < height; x++){
			order[x] = omp_get_thread_num();

			for(int y=0; y < width; y++)
			{
				int grad_x = 0;
				int grad_y = 0;
				int grad = 0;

				if (x == 0 || x == (height - 1) || y == 0 || y == (width - 1))
					grad = 0;
				else {
					/* Gradient calculation in X Dimension */
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++)
						{
							grad_x += (inputImage[x + i][y + j] * maskX[i+1][j+1]);
						}
					}

					/* Gradient calculation in Y Dimension */
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++)
						{
							grad_y += (inputImage[x + i][y + j] * maskY[i+1][j+1]);
						}
					}
					/* Gradient magnitude */
					grad = (int) sqrt( (grad_x * grad_x) + (grad_y * grad_y) );
				}

				outputImage[x][y] = clamp(grad, 0, 255);

			}
		}
	}
}


// Part A2
void perwitt_dynamic()
{
	#pragma omp parallel shared(inputImage,outputImage,chunkSize,order)
	{
		#pragma omp for schedule(dynamic,chunkSize) nowait
		for(int x = 0; x < height; x++){
			order[x] = omp_get_thread_num();

			for(int y = 0; y < width; y++)
			{
				int grad_x = 0;
				int grad_y = 0;
				int grad = 0;

				if (x == 0 || x == (height - 1) || y == 0 || y == (width - 1))
					grad = 0;
				else {
					/* Gradient calculation in X Dimension */
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++) {
							grad_x += (inputImage[x + i][y + j] * maskX[i+1][j+1]);
						}
					}
					/* Gradient calculation in Y Dimension */
					for (int i = -1; i <= 1; i++) {
						for (int j = -1; j <= 1; j++) {
							grad_y += (inputImage[x + i][y + j] * maskY[i+1][j+1]);
						}
					}
					/* Gradient magnitude */
					grad = (int) sqrt( (grad_x * grad_x) + (grad_y * grad_y) );
				}

				outputImage[x][y] = clamp(grad, 0, 255);

			}
		}
	}

}



int main(int argc, char* argv[])
{
		/* 3x3 Prewitt mask for X Dimension. */
		maskX[0][0] = +1; maskX[0][1] = 0; maskX[0][2] = -1;
		maskX[1][0] = +1; maskX[1][1] = 0; maskX[1][2] = -1;
		maskX[2][0] = +1; maskX[2][1] = 0; maskX[2][2] = -1;

		/* 3x3 Prewitt mask for Y Dimension. */
		maskY[0][0] =  1; maskY[0][1] =  1; maskY[0][2] =  1;
		maskY[1][0] =  0; maskY[1][1] =  0; maskY[1][2] =  0;
		maskY[2][0] = -1; maskY[2][1] = -1; maskY[2][2] = -1;


		if(argc != 6)
    {
        std::cout << "Arguments: Input image file directory  Output image file directory  Chunk size  ThreadCount  A1orA2" << std::endl;
        return 0;
    }


		//input file
    std::ifstream file(argv[1]);
    if(!file.is_open())
    {
        std::cout << "Could not open file " << argv[1] << std::endl;
        return 0;
    }

		threadsCount = std::atoi(argv[4]);
    chunkSize = std::atoi(argv[3]);

    std::cout << "Processing " << argv[1] << " using " << threadsCount << " OpenMP threads\n" << std::endl;



		std::string workString;


    // Removes comments and check image format
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            if( workString.at(1) != '2' ){
                std::cout << "Input image is not valid" << std::endl;
                return 0;
            } else {
                break;
            }
        } else {
            continue;
        }
    }


    // Get image size
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            int n;
            stream >> n;
            width = n;
            stream >> n;
            height = n;
            break;
        } else {
            continue;
        }
    }



    // Fill input image
    int pixelValue;
    for( int i = 0; i < height; i++ )
    {
        if( std::getline(file,workString) && workString.at(0) != '#' )
				{
            std::stringstream stream(workString);
            for( int j = 0; j < width; j++ )
						{
                if( !stream )
                    break;
                stream >> pixelValue;
                inputImage[i][j] = pixelValue;
            }
        } else {
            continue;
        }
    }


    // proccess image
    order = new int[height];

    omp_set_num_threads(threadsCount);

    double time_static, time_dynamic;

    std::string part = argv[5];
    if( !part.compare("A1") )
    {
        time_static = omp_get_wtime();
        perwitt_static();
        time_static = omp_get_wtime() - time_static;
        std::cout << "Part A1 time taken: " << time_static << std::endl;
    }
		else if ( !part.compare("A2") )
		{
        time_dynamic = omp_get_wtime();
        perwitt_dynamic();
        time_dynamic = omp_get_wtime() - time_dynamic;
        std::cout << "Part A2's time taken: " << time_dynamic << std::endl;
    }else{
				std::cout << "Invalid Part" << std::endl;
				return 0;
		}


    //output file
		std::ofstream outfile(argv[2]);
		if (outfile.is_open())
		{
			outfile << "P2" << "\n" << width << " " << height << "\n" << 255 << "\n";
			for( int i = 0; i < height; i++ ){
          for( int j = 0; j < width; j++ )
					{
							outfile << outputImage[i][j] << " ";
          }
          outfile << std::endl;
      }
			outfile.close();
		} else {
		    std::cout << "Could not open output file " << argv[2] << std::endl;
		    return 0;
		}


		int current_task = order[0];
    printf("Task %d ­-> Processing Chunk starting at Row %d\n", current_task, 0);

    for(int i = 1; i < height; i++)
		{
    	if (current_task != order[i])
			{
    		current_task = order[i];
    		printf("Task %d ­-> Processing Chunk starting at Row %d\n", current_task, i);
    	}

    }



    return 0;
}











// ImplementationC.cpp
// compile: g++ -o main -fopenmp "C:\Users\User1\Documents\openmp\ImplementationC.cpp"
// run: ./main

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
