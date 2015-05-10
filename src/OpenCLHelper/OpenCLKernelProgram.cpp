/*
 * OpenCLKernelProgram.cpp
 *
 *  Created on: May 7, 2015
 *      Author: Mikael
 */

#include "OpenCLKernelProgram.h"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

namespace ATML
{
namespace Helper
{

OpenCLKernelProgram::OpenCLKernelProgram()
{

}

OpenCLKernelProgram::~OpenCLKernelProgram()
{

}

string OpenCLKernelProgram::GetTextFromPath(const string& path)
{
	ifstream file(path);
	stringstream stringStream;
	string temp;
	while (getline(file, temp))
		stringStream << temp << endl;

	file.close();

	return stringStream.str();
}

} /* namespace Helper */
} /* namespace ATML */
