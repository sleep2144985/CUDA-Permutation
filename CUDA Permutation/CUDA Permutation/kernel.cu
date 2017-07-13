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

__device__ bool Compare(int* set, int* payTable, int size) {
    int Any = 0;
    for (int i = 0; i < size; i++) {
        if (set[i] != -1 && payTable[i] > 0) {
            // ordinary compare
            if (set[i] != payTable[i]) {
                return false;
            }
        } else if (set[i] != -1 && payTable[i] == -1) {
            // any
            if (Any == 0) {
                Any = set[i];
            } else {
                if (set[i] != Any) {
                    return false;
                }
            }
        }
    }

    return true;
}

// 設定每個kernel的亂數種子
__global__ void SetupCurand(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}
// 跑模擬
__global__ void Simulate(curandState *states, const int colunmSize, const int rowSize, int* reelSets, const int reelSetSize, int* payTable, int payTableSize, size_t runTimes, size_t* hitTimes, size_t* noHitTimes, const size_t NUM_OF_THREAD) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[idx];
    int* set;
    set = (int*) malloc(colunmSize * rowSize * sizeof(int));
    for (int t = 0; t < runTimes; t++) {
        for (int col = 0; col < colunmSize; col++) {
            unsigned int rand = curand(&localState) % reelSetSize;
            for (int row = 0; row < rowSize; row++) {
                set[row + col * rowSize] = reelSets[(rand + row) % reelSetSize];
            }
        }
        // 核對盤面
        //bool hit = false;
        for (int n = 0; n < payTableSize; n++) {
            if (Compare(set, (payTable + colunmSize * rowSize * n), colunmSize * rowSize)) {
                //hit = true;
                hitTimes[idx + n * NUM_OF_THREAD] += 1;
            }
        }
        // 多紀錄一個未中獎在hitTimes的最前面區塊
        /*if (!hit) {
            noHitTimes[idx] += 1;
        }*/

        states[idx] = localState;
    }
    free(set);
};

int main(int argc, char** argv) {
    const unsigned int RUN_TIMES = 1000;

    // 加入 Console 參數
    if (argc != 3) { printf(".exe [input file] [output file]\n"); return 1; }
    string intputPath = argv[1];
    string outputPath = argv[2];

    // 設定輸入輸出檔案
    InputCSV inputFile(intputPath);
    OutputCSV outputFile(outputPath);

    // 計時開始
    unsigned long cStart = clock();

    // 模擬 Column*Row 的盤面.
    const int COLUMN_SIZE = inputFile.getPermutationColumnSize();
    const int REEL_ROW_SIZE = inputFile.getReelRowSize();

    // Symbols
    const string *SYMBOLS = inputFile.getPermutationElements();
    const int SYMBOLS_SIZE = inputFile.getPermutationElementsCount();

    // Stops.(停止點 = Reel上的元素)
    const int* STOPS = inputFile.getReelSet();
    const int STOPS_SIZE = inputFile.getReelSetSize();

    // Pay Table
    const int* PAY_TABLE = inputFile.getPayTable();
    // Size of pay table(Element count).
    const int PAY_TABLE_SIZE = inputFile.getPayTableSize();

    // PAY_TABLE 裡變數的數量 = sizeof(PAY_TABLE) / sizeof(int)
    const int PAY_TABLE_REAL_SIZE = PAY_TABLE_SIZE * COLUMN_SIZE * REEL_ROW_SIZE;

    //---------------------Begin of cuda-----------------------------
    size_t *hitTimes;
    size_t *host_hitTimes;
    size_t *dev_hitTimes;

    size_t *host_noHitTimes;
    size_t *dev_noHitTimes;


    int* dev_reelSets;
    int* dev_winningSets;


    // 設定 thread & block.
    size_t threads = 10;
    size_t blocks = 10;

    size_t NumOfThread = blocks * threads, kernelRunTimes = ceil(RUN_TIMES / NumOfThread);
    printf("Total times: %d\nBlock count: %d\nThread count: %d\nKernelRunTimes: %d\n", RUN_TIMES, blocks, threads, kernelRunTimes);

    // 配置Host memory.
    hitTimes = (size_t*) malloc(PAY_TABLE_SIZE * sizeof(size_t));
    host_hitTimes = (size_t*) malloc(NumOfThread * PAY_TABLE_SIZE * sizeof(size_t));
    //host_noHitTimes = (size_t*) malloc(NumOfThread * sizeof(size_t));

    // 配置Device memory.
    cudaMalloc((void**) &dev_hitTimes, NumOfThread * PAY_TABLE_SIZE * sizeof(size_t));
    cudaMalloc((void**) &dev_noHitTimes, sizeof(size_t));

    // Declare reel sets.
    cudaMalloc((void**) &dev_reelSets, STOPS_SIZE * sizeof(int));
    cudaMemcpy(dev_reelSets, STOPS, STOPS_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Declare pay table.
    cudaMalloc((void**) &dev_winningSets, PAY_TABLE_SIZE * sizeof(size_t));
    cudaMemcpy(dev_winningSets, PAY_TABLE, PAY_TABLE_SIZE * sizeof(size_t), cudaMemcpyHostToDevice);

    // Setup random seed for each threads.
    curandState* devStates;
    cudaMalloc(&devStates, NumOfThread * sizeof(curandState));
    SetupCurand <<<blocks, threads>>> (devStates, time(NULL));

    // Simulate.
    Simulate <<<blocks, threads>>> (devStates, COLUMN_SIZE, REEL_ROW_SIZE, dev_reelSets, STOPS_SIZE, dev_winningSets, PAY_TABLE_SIZE, kernelRunTimes, dev_hitTimes, dev_noHitTimes, NumOfThread);

    // Copy device memory to host.
    cudaMemcpy(host_hitTimes, dev_hitTimes, NumOfThread * PAY_TABLE_SIZE * sizeof(size_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(host_noHitTimes, dev_noHitTimes, NumOfThread * sizeof(size_t), cudaMemcpyDeviceToHost);


    //釋放Memory.
    cudaFree(dev_reelSets);
    cudaFree(dev_winningSets);
    cudaFree(dev_hitTimes);
    cudaFree(dev_noHitTimes);

    //---------------------End of cuda-----------------------------
    // 算no hit.
    size_t noHitTimes = 0;
    /*for (size_t i = 0; i < NumOfThread; i++) {
        noHitTimes += host_hitTimes[i];
    }*/
    // 算hit.
    size_t totalHitTimes = 0;
    for (size_t i = 0; i < PAY_TABLE_SIZE; i++) {
        hitTimes[i] = 0;
        for (size_t t = 0; t < NumOfThread; t++) {
            printf("%u\n", host_hitTimes[i * NumOfThread + t]);
            hitTimes[i] += host_hitTimes[i * NumOfThread + t];
        }
        // 紀錄總hit次數
        //totalHitTimes += hitTimes[i];
    }
    // 計時完了
    unsigned long spendTime = clock() - cStart;

    // Console print.
    printf("CUDA run %lu ms.\n", spendTime);
    printf("Output to %s... \n", outputPath.c_str());

    // 輸出
    outputFile.WriteTitle(blocks, threads, RUN_TIMES, spendTime, STOPS_SIZE, COLUMN_SIZE, REEL_ROW_SIZE, totalHitTimes, (double)totalHitTimes / RUN_TIMES);

    outputFile.WriteHitFreq("No hit", noHitTimes, (double) noHitTimes / RUN_TIMES);

    // Output hit frequency to file.
    for (int i = 0; i < PAY_TABLE_SIZE; i++) {
        outputFile.WriteHitFreq(inputFile.getPayTableFileName(i), hitTimes[i], (double) hitTimes[i] / RUN_TIMES);
    }



    outputFile.Close();
    delete[] host_hitTimes;
    //delete[] host_noHitTimes;
    delete[] hitTimes;

    printf("Finish.\n");
    return 0;
}
