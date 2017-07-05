#include "OutputCSV.h"



OutputCSV::OutputCSV() {
}

OutputCSV::OutputCSV(string path) {
    fout.open(path);
}

OutputCSV::~OutputCSV() {
}

void OutputCSV::WriteTitle(int blockCount, int threadCount, unsigned int targetRunTimes, unsigned int realRunTimes, unsigned long clock, int elementCount, int length,int reelRowSize) {
    this->fout << "Block计秖," << blockCount << endl;
    this->fout << "Thread计秖," << threadCount << endl;
    this->fout << "箇戳家览Ω计," << targetRunTimes << endl;
    this->fout << "龟悔家览Ω计," << realRunTimes << endl;
    this->fout << "磅︽丁(ms)," << clock << endl;
    this->fout << "じ计," << elementCount << endl;
    this->fout << "锣絃计," << length << endl;
	this->fout << "瞷计," << reelRowSize << endl;
    this->fout << "い贱舱,瞷Ω计,瞷诀瞯" << endl;
}

void OutputCSV::WriteWinningRate(string name,size_t count,double percentage){
	this->fout << name << "," << count << "," << percentage << endl;
}

void OutputCSV::WriteRowData(string permutation, int count, double percentage) {
    this->fout << permutation << "," << count << "," << percentage << "%" << endl;
}

void OutputCSV::Close() {
    if (fout.is_open()) {
        fout.close();
    }
}
