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
	vector<string> reelSet;
	vector<string> winingSets;

	// Get titles [Symbol/ Reel count/ Reel/ Wining sets]
	getline(fin,inputLine);
	// Read line move to nextt line.
	inputLineCount++;
	// Split "," charactor.
	titles = split(inputLine,",");


	while(!fin.eof()){
		getline(fin,inputLine);
		// Split "," charactor.
		// If this line dont contains elements break.
		if(inputLine.empty()){
			break;
		}
		vector<string> elements = split(inputLine,",");
		
		// First column is symbol.
		symbols.push_back(elements[0]);
		// Second column is reel count.
		// Only read it at first line.
		if(inputLineCount == 1){
			reelCount = elements[1];
		}
		// Third column is reel set.
		reelSet.push_back(elements[2]);
		// Fourth column is wining set.
		winingSets.push_back(elements[3]);
		// Move to next line.
		inputLineCount++;
	}
	// ¦s¨ìmember variable.
	this->_permutationElementsCount = symbols.size();
	this->_permutationElements = new string[this->_permutationElementsCount];
	copy(symbols.begin(),symbols.end(),this->_permutationElements);

	this->_permutationLength = atoi(reelCount.c_str());

	printf("reel count: %d\n",this->_permutationLength);
	for(int i = 0;i < this->_permutationElementsCount;i++){
		std::cout << this->_permutationElements[i] << std::endl;
	}
    
    fin.close();
}


InputCSV::~InputCSV() {
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
