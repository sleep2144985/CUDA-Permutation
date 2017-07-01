#pragma once
#include<string>
using namespace std;

class InputCSV {
private:
    string _path;
public:
    InputCSV();
    InputCSV(string path);
    ~InputCSV();
    
    // 取得排列組合的長度(int)
    int ReadPermutationLength();
    // 取得排列組合的元素(string array)
    string* ReadPermutationElements();
};

