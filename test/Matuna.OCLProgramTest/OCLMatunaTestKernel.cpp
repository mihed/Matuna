/*
* OCLMatunaTestKernel.cpp
*
*  Created on: Jun 10, 2015
*      Author: Mikael
*/

#include "OCLMatunaTestKernel.h"
#include "Matuna.OCLHelper/OCLProgram.h"

OCLMatunaTestKernel::OCLMatunaTestKernel()
{
	name = "DivideByScalarKernel";
	string path = OCLProgram::DefaultSourceLocation + "MatunaTestKernel.cl";
	includePaths.push_back(OCLProgram::DefaultSourceLocation);
	sourcePaths.push_back(path);
	unordered_map<string, string> subsMap;
	subsMap.insert(make_pair("MATUNA_TEST_DEFINE2", "100"));
	unordered_set<string> defineSet;
	defineSet.insert("MATUNA_TEST_DEFINE");
	defines.insert(make_pair(path, defineSet));
	subs.insert(make_pair(path, subsMap));
}

OCLMatunaTestKernel::~OCLMatunaTestKernel()
{

}

string OCLMatunaTestKernel::Name() const
{
	return name;
}

const vector<size_t>& OCLMatunaTestKernel::GlobalWorkSize() const
{
	return globalWorkSize;
}

const vector<size_t>& OCLMatunaTestKernel::LocalWorkSize() const
{
	return localWorkSize;
}

vector<string> OCLMatunaTestKernel::GetIncludePaths() const
{
	return includePaths;
}

vector<string> OCLMatunaTestKernel::GetSourcePaths() const 
{
	return sourcePaths;
}

unordered_map<string,unordered_map<string, string>> OCLMatunaTestKernel::GetDefineSubstitutes() const 
{
	return subs;
}

unordered_map<string, unordered_set<string>> OCLMatunaTestKernel::GetDefines() const
{
	return defines;
}

