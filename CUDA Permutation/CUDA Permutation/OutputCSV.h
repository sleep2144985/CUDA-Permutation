#pragma once
#include<string>
using namespace std;

class OutputCSV {
private:
    string _path;
public:
    OutputCSV();
    OutputCSV(string path);
    ~OutputCSV();

    // 輸出資料(參數自己加)
    void WriteData();
};

