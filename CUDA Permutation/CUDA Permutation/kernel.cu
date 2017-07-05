#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <string>
#include<iostream>

#include "InputCSV.h"
#include "OutputCSV.h"


using namespace std;

__device__ bool Compare(int* set,int* winningSet,int size){
	int Any = 0;
	for(int i = 0;i < size;i++){
		if(winningSet[i] > 0){
			// ordinary compare
			if(set[i] != winningSet[i]){
				return false;
			}
		} else if(winningSet[i] == -1){
			// any
			if(Any == 0){
				Any = set[i];
			} else{
				if(set[i] != Any){
					return false;
				}
			}
		}
	}

	return true;
}

// 設定每個kernel的亂數種子
__global__ void SetupCurand(curandState *state,unsigned long seed){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed,idx,0,&state[idx]);
}
// 跑模擬
__global__ void Simulate(curandState *states,int colunmSize,int rowSize,int* reelSets,int reelSetSize,int* winningSets,int winningSetSize,int times,size_t* winningSetCount,size_t* count){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = states[idx];
	int* set;
	set = (int*)malloc(colunmSize*rowSize * sizeof(int));

	float RANDOM = curand_uniform(&localState);

	for(int col = 0;col < colunmSize;col++){
		unsigned int rand = curand(&localState) % reelSetSize;
		for(int row = 0;row < rowSize;row++){
			set[col + row*rowSize] = reelSets[(rand + row) % reelSetSize];
		}
	}

	for(int n = 0;n < winningSetSize; n++){
		if(Compare(set,(winningSets + colunmSize*rowSize*n),colunmSize*rowSize)){
			winningSetCount[n] += 1;
		}
	}
	count[0] += 1;
	free(set);
	states[idx] = localState;
};

int main(int argc,char** argv){
	// 加入參數
	//if(argc != 3){ printf(".exe [input file] [output file]\n"); return 1; }
	string intputPath = "input.csv";//argv[1];
	string outputPath = "output.csv";//argv[2];

	unsigned long cStart = clock();
	InputCSV inputFile(intputPath);
	OutputCSV outputFile(outputPath);
	const unsigned int RUN_TIMES = 100000;
	const int LENGTH = inputFile.getPermutationLength();
	const int REEL_ROW_SIZE = inputFile.getReelRowSize();

	const string *ELEMENTS = inputFile.getPermutationElements();
	const int ELEMENTS_SIZE = inputFile.getPermutationElementsCount();

	const int* REEL_SETS = inputFile.getReelSet();
	const int REEL_SET_SIZE = inputFile.getReelSetSize();

	const int WINNING_SET_SIZE = inputFile.getWinningSetSize();
	const int* WINNING_SETS = inputFile.getWinningSets();
	const int SET_SIZE = WINNING_SET_SIZE / LENGTH / REEL_ROW_SIZE;

	//---------------------Begin of cuda-----------------------------
	size_t *winningSetCount;
	size_t *dev_winningSetCount;
	size_t *Count;
	size_t *dev_Count;

	int* dev_reelSets;
	int* dev_winningSets;


	// 設定 thread & block.
	unsigned int threads = 1000;
	unsigned int blocks = 10;

	unsigned int NumOfThread = blocks*threads, kernelRunTimes = ceil(RUN_TIMES / NumOfThread);
	printf("Total times: %d\nBlock count: %d\nThread count: %d\nKernelRunTimes: %d\n",RUN_TIMES,blocks,threads,kernelRunTimes);

	// 配置Host memory.
	winningSetCount = (size_t*)malloc(SET_SIZE * sizeof(size_t));
	Count = (size_t*)malloc(SET_SIZE * sizeof(size_t));


	// 配置Device memory.
	cudaMalloc((void**)&dev_winningSetCount,SET_SIZE * sizeof(size_t));

	cudaMalloc((void**)&dev_Count,SET_SIZE * sizeof(size_t));

	cudaMalloc((void**)&dev_reelSets,REEL_SET_SIZE * sizeof(int));

	cudaMalloc((void**)&dev_winningSets,SET_SIZE * sizeof(int));

	cudaMemcpy(dev_reelSets,REEL_SETS,REEL_SET_SIZE*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_winningSets,WINNING_SETS,SET_SIZE *sizeof(int),cudaMemcpyHostToDevice);

	curandState* devStates;
	cudaMalloc(&devStates,NumOfThread * sizeof(curandState));

	SetupCurand <<<blocks,threads >>>(devStates,time(NULL));

	Simulate <<<blocks,threads >>>(devStates,LENGTH,REEL_ROW_SIZE,dev_reelSets,REEL_SET_SIZE,dev_winningSets,SET_SIZE,kernelRunTimes,dev_winningSetCount,dev_Count);

	// Copy device memory to host.
	cudaMemcpy(winningSetCount,dev_winningSetCount,SET_SIZE * sizeof(size_t),cudaMemcpyDeviceToHost);
	cudaMemcpy(Count,dev_Count,SET_SIZE * sizeof(size_t),cudaMemcpyDeviceToHost);


	//釋放Memory.
	cudaFree(dev_reelSets);
	cudaFree(dev_winningSets);
	cudaFree(dev_winningSetCount);
	cudaFree(dev_Count);

	//---------------------End of cuda-----------------------------

	unsigned long cEnd = clock();
	printf("CUDA run %lu ms.\n",cEnd - cStart);

	printf("Output to %s... \n",outputPath.c_str());

	size_t total = 0;
	for(int i = 0;i < SET_SIZE;i++){
		total += Count[i];
	}
	// 輸出
	outputFile.WriteTitle(blocks,threads,RUN_TIMES,total,cEnd - cStart,ELEMENTS_SIZE,LENGTH,REEL_ROW_SIZE);

	//output winning rate ot csv file.
	for(int i = 0;i < SET_SIZE;i++){
		//[TEMP]
		outputFile.WriteWinningRate(inputFile.getWinningSetName(i),winningSetCount[i],((double)winningSetCount[i] / total));
	}



	outputFile.Close();

	delete[] winningSetCount;

	printf("Finish.\n");
	system("PAUSE");
	return 0;
}