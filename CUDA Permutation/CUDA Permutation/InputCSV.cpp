#include "InputCSV.h"

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

    // 讀第一行.
    getline(fin, inputLine);
    // Split "," charactor.
    vector<string> elements = split(inputLine, ",");
    // 存到member variable.
    this->_permutationElementsCount = elements.size() - 1;
    this->_permutationElements = new string[this->_permutationElementsCount];
    copy(elements.begin() + 1, elements.end(), this->_permutationElements);

    // 讀第二行.
    getline(fin, inputLine);
    string lengthStr = split(inputLine, ",")[1];
    
    this->_permutationLength = atoi(lengthStr.c_str());
    
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
