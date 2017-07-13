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
    void WriteTitle(int blockCount, int threadCount, size_t realRunTimes, unsigned long clock, int stopsSize, int columnSize, int rowSize, size_t totalHitTimes, double totalHitFreq);
	void WriteHitFreq(string name,size_t count,double percentage);

    void Close();
};

