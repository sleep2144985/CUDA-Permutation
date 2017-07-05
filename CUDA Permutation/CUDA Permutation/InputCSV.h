#pragma once
#include<string>
#include<fstream>
#include<vector>

using namespace std;

class InputCSV {
private:
    int _permutationLength;
	int _permutationRowSize;

	int _permutationReelSetCount;
	int* _permutationReelSets;

    int _permutationElementsCount;
    string* _permutationElements;

	int _permutationWiningSetCount;
	string* _permutationWiningSets;

	
	vector<string> OpenWiningSetFile(vector<string>&);

public:
    InputCSV();
    InputCSV(string path);
    ~InputCSV();
    
    // 取得排列組合的長度(int)
    int getPermutationLength();
    // 取得排列組合的元素的總數(int)
    int getPermutationElementsCount();
    // 取得排列組合的元素(string array)
    string* getPermutationElements();
};

