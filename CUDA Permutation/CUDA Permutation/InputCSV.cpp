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
	vector<int> reelSet;
	string         rowSize;
	vector<string> winingSetFiles;
	vector<string> winingSets;

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
		if(!elements[1].empty()){
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
	this->_permutationLength = atoi(reelCount.c_str());

	// Reel setting.
	// -1 for any symbol
	this->_permutationReelSetCount = reelSet.size();
	this->_permutationReelSets = new int[this->_permutationReelSetCount];
	copy(reelSet.begin(),reelSet.end(),this->_permutationReelSets);

	// Row size.
	this->_permutationRowSize = atoi(rowSize.c_str());

	// Winning sets.
	winingSets = OpenWiningSetFile(winingSetFiles);
	this->_permutationWiningSetCount = winingSets.size();
	this->_permutationWiningSets = new string[this->_permutationWiningSetCount];
	copy(winingSets.begin(),winingSets.end(),this->_permutationWiningSets);

    fin.close();
}

vector<string> InputCSV::OpenWiningSetFile(vector<string>& files){

}


InputCSV::~InputCSV() {
	// delete
	delete[] _permutationReelSets;
	delete[] _permutationElements;
	delete[] _permutationWiningSets;
}

int InputCSV::getPermutationLength() {
    return this->_permutationLength;
}

int InputCSV::getPermutationElementsCount() {
    return this->_permutationElementsCount;
}

string* InputCSV::getPermutationElements() {
    return this->_permutationElements;
}
