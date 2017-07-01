#include "OutputCSV.h"



OutputCSV::OutputCSV() {
}

OutputCSV::OutputCSV(string path) {
    fout.open(path);
}

OutputCSV::~OutputCSV() {
}

void OutputCSV::WriteTitle(int blockCount, int threadCount, int targetRunTimes, int realRunTimes, int clock, int elementCount, int length, int permutationCount) {
    this->fout << "Block數量," << blockCount << endl;
    this->fout << "Thread數量," << threadCount << endl;
    this->fout << "預期模擬次數," << targetRunTimes << endl;
    this->fout << "實際模擬次數," << realRunTimes << endl;
    this->fout << "執行時間(ms)," << clock << endl;
    this->fout << "元素個數," << elementCount << endl;
    this->fout << "轉盤個數," << length << endl;
    this->fout << "排列組合總數," << permutationCount << endl;
    this->fout << "排列組合,出現次數,出現機率" << endl;
}

void OutputCSV::WriteRowData(string permutation, int count, double percentage) {
    this->fout << permutation << "," << count << "," << percentage << "%" << endl;
}

void OutputCSV::Close() {
    if (fout.is_open()) {
        fout.close();
    }
}
