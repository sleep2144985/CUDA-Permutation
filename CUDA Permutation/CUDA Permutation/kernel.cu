#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <string>

#include "InputCSV.h"
#include "OutputCSV.h"


using namespace std;


// 設定每個kernel的亂數種子
__global__ void SetupCurand(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}
// 跑模擬
__global__ void Simulate(curandState *states, int length, int elementCount, int times, unsigned int* countOfPermutation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[idx];
    float RANDOM = curand_uniform(&localState);
    for (int k = 0; k < times; ++k) {
        unsigned int index = 0;
        for (int i = 0; i < length; ++i) {
            unsigned int rand = curand(&localState);
            unsigned int power = 1;
            for (int j = 0; j < i; j++) { power *= elementCount; }
            index += (rand % elementCount) * power;
        }
        countOfPermutation[index] += 1;
    }
    states[idx] = localState;
};

int main(int argc, char** argv) {
    // 加入參數
    if (argc != 3) { printf(".exe [input file] [output file]\n"); return 1; }
    string intputPath = argv[1];
    string outputPath = argv[2];

    unsigned long cStart = clock();
    InputCSV inputFile(intputPath);
    OutputCSV outputFile(outputPath);
    const unsigned int RUN_TIMES = 100000000;
    const int LENGTH = inputFile.getPermutationLength();
    const string *ELEMENTS = inputFile.getPermutationElements();
    const int ELEMENTS_SIZE = inputFile.getPermutationElementsCount();

    size_t *countOfPermutation;
    size_t *dev_countOfPermutation;
    size_t PERMUTATION_COUNT = pow(ELEMENTS_SIZE, LENGTH);
    printf("總共有%u種排列組合\n", PERMUTATION_COUNT);

    // 配置Host memory.
    countOfPermutation = (size_t*) malloc(PERMUTATION_COUNT * sizeof(size_t));
    // 配置Device memory.
    cudaMalloc((void**) &dev_countOfPermutation, PERMUTATION_COUNT * sizeof(size_t));

    unsigned int threads = 10;
    unsigned int blocks = 1000;

    unsigned int NumOfThread = blocks*threads, kernelRunTimes = ceil(RUN_TIMES / NumOfThread);
    printf("Total times: %d\nBlock count: %d\nThread count: %d\nKernelRunTimes: %d\n", RUN_TIMES, blocks, threads, kernelRunTimes);

    curandState* devStates;
    cudaMalloc(&devStates, NumOfThread * sizeof(curandState));
    SetupCurand<<<blocks, threads>>>(devStates, time(NULL));

    Simulate<<<blocks, threads>>>(devStates, LENGTH, ELEMENTS_SIZE, kernelRunTimes, dev_countOfPermutation);

    // Copy device memory to host.
    cudaMemcpy(countOfPermutation, dev_countOfPermutation, PERMUTATION_COUNT * sizeof(size_t), cudaMemcpyDeviceToHost);
    
    unsigned long cEnd = clock();
    printf("CUDA run %lu ms.\n", cEnd - cStart);
    
    printf("Output to %s... \n", outputPath.c_str());
    // 算總共跑了幾次
    size_t total = 0;
    for (size_t i = 0; i < PERMUTATION_COUNT; i++) {
        total += countOfPermutation[i];
    }

    // 輸出
    outputFile.WriteTitle(blocks, threads, RUN_TIMES, total, cEnd - cStart, ELEMENTS_SIZE,LENGTH, PERMUTATION_COUNT);
    for (size_t i = 0; i < PERMUTATION_COUNT; i++) {
        string symbols;
        for (size_t j = 0; j < LENGTH; j++) {
            size_t index = (size_t)((double)i / pow(ELEMENTS_SIZE, j)) % ELEMENTS_SIZE;
            if (j == 0) { symbols = ELEMENTS[index] + symbols; }
            else symbols = ELEMENTS[index] + "-" + symbols;
        }
        //printf("#%u %s appear %u times (%0.3f %%)\n", i, symbols.c_str(), countOfPermutation[i], ((double)countOfPermutation[i]/total) * 100);
        outputFile.WriteRowData(symbols, countOfPermutation[i], (double) countOfPermutation[i] / total * 100);
    }
    outputFile.Close();
    //釋放Memory.
    cudaFree(dev_countOfPermutation);
    delete[] countOfPermutation;

    printf("Finish.\n");
    return 0;
}