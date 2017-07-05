#include "InputCSV.h"
#include<iostream>

vector<string> split(const string &source, const string &delim) {
    vector<string> ans;
    size_t begin_pos = 0, end_pos = source.find(delim); // locate the first delimiter in string
    while (end_pos != string::npos) {
        ans.push_back(source.substr(begin_pos, end_pos - begin_pos)); // extract the sub-string before current delimiter
        begin_pos = end_pos + delim.size();
        end_pos = source.find(delim, begin_pos);  // locate the next delimiter in string
    }
    ans.push_back(source.substr(begin_pos, end_pos - begin_pos));  // extract the last sub-string
    return ans;
}

InputCSV::InputCSV() {
}

InputCSV::InputCSV(string path) {
    ifstream fin(path);
    string inputLine;
	unsigned int inputLineCount = 0;

	// CSV elements
	vector<string> titles;
	vector<string> symbols;
	string         reelCount;
	vector<int>    reelSet;
	string         rowSize;
	vector<string> winingSetFiles;
	//[Need Improve Data Structure] 
	vector<int>    winingSets;

	// Get titles [Symbol/ Reel count/ Reel/ Row size/ Wining sets]
	getline(fin,inputLine);
	// Read line move to nextt line.
	inputLineCount++;
	// Split "," charactor.
	titles = split(inputLine,",");


	while(!fin.eof()){
		getline(fin,inputLine);
		
		// If this line doesnt contains element then break.
		if(inputLine.empty()){
			break;
		}

		// Split "," charactor.
		vector<string> elements = split(inputLine,",");
		
		// First column is symbol.
		if(!elements[0].empty()){
			symbols.push_back(elements[0]);
		}
		
		// Only read it at first line.
		if(inputLineCount == 1){
			// Second column is reel count.
			reelCount = elements[1];
			//Fourth column is row size.
			rowSize = elements[3];
		}

		// Third column is reel set.
		if(!elements[2].empty()){
			reelSet.push_back(atoi(elements[2].c_str()));
		}
		

		// Fifth column is wining set.
		if(!elements[4].empty()){
			winingSetFiles.push_back(elements[4]);
		}
		
		// Move to next line.
		inputLineCount++;
	}
	// ¦s¨ìmember variable.

	// Symbols.
	this->_permutationElementsCount = symbols.size();
	this->_permutationElements = new string[this->_permutationElementsCount];
	copy(symbols.begin(),symbols.end(),this->_permutationElements);

	// Reel Count.
	this->_permutationColumnSize = atoi(reelCount.c_str());

	// Reel setting.
	// -1 for any symbol
	this->_permutationReelSetCount = reelSet.size();
	this->_permutationReelSets = new int[this->_permutationReelSetCount];
	copy(reelSet.begin(),reelSet.end(),this->_permutationReelSets);

	// Row size.
	this->_permutationRowSize = atoi(rowSize.c_str());
	//Winning sets name.
	this->_winningSetNames = new string[winingSetFiles.size()];
	copy(winingSetFiles.begin(),winingSetFiles.end(),_winningSetNames);

	// Winning sets.
	if(!OpenWiningSetFile(winingSetFiles,winingSets)){
		// open faliure.
	}
	this->_permutationWiningSetCount = winingSets.size();
	this->_permutationWiningSets = new int[this->_permutationWiningSetCount*this->_permutationRowSize*this->_permutationColumnSize];
	copy(winingSets.begin(),winingSets.end(),this->_permutationWiningSets);

    fin.close();
}

bool InputCSV::OpenWiningSetFile(vector<string>& files,vector<int>& winingSets){
	fstream fin;
	// open every files and load into WiningSets.
	for(string& file : files){
		string set;
		vector<vector<string>> tempName;
		fin.open(file);
		if(fin.fail()){
			return false;
		}
		string inputLine;
		while(!fin.eof()){
			getline(fin,inputLine);
			tempName.push_back(split(inputLine,","));
		}

		for(int i = 0;i < tempName[0].size();i++){
			for(int j = 0;j < tempName.size();j++){
				winingSets.push_back(atoi(tempName[j][i].c_str()));
			}
		}
		fin.close();
	}
	return true;
}


InputCSV::~InputCSV() {
	// delete
	delete[] _permutationReelSets;
	delete[] _permutationElements;
	delete[] _permutationWiningSets;
	delete[] _winningSetNames;
}

int InputCSV::getPermutationLength() {
    return this->_permutationColumnSize;
}

int InputCSV::getPermutationElementsCount() {
    return this->_permutationElementsCount;
}

string* InputCSV::getPermutationElements() {
    return this->_permutationElements;
}

int InputCSV::getWinningSetSize(){
	return _permutationWiningSetCount;
}

string InputCSV::getWinningSetName(int i){
	return _winningSetNames[i];
}

// get row size.
int InputCSV::getReelRowSize(){
	return _permutationRowSize;
}

// get reel set size
int InputCSV::getReelSetSize(){
	return _permutationReelSetCount;
}
// get reel set.
int* InputCSV::getReelSet(){
	return _permutationReelSets;
}
// get winning sets
int* InputCSV::getWinningSets(){
	return _permutationWiningSets;
}
