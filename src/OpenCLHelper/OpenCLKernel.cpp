/*
 * OpenCLKernel.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */

#include "OpenCLKernel.h"
#include <iostream>
#include <sstream>
#include <fstream>

namespace ATML
{
namespace Helper
{

int OpenCLKernel::instanceCounter = 0;

OpenCLKernel::OpenCLKernel()
{
	instanceCounter++;
}

OpenCLKernel::~OpenCLKernel()
{

}

string OpenCLKernel::GetTextFromPath(string path)
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
