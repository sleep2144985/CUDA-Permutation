#pragma once
#include<string>
#include<fstream>
#include<iomanip>
using namespace std;

class OutputCSV {
private:
    ofstream fout;
public:
    OutputCSV();
    OutputCSV(string path);
    ~OutputCSV();

    // Write.
    void WriteTitle(int blockCount, int threadCount, unsigned int targetRunTimes, unsigned int realRunTimes, unsigned long clock, int elementCount, int length, unsigned int permutationCount);
    void WriteRowData(string permutation, int count, double percentage);

    void Close();
};

