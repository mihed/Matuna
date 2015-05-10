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
	kernel = nullptr;
	context = nullptr;
	kernelSet = false;
}

OpenCLKernel::~OpenCLKernel()
{

}

void OpenCLKernel::SetOCLKernel(cl_kernel kernel)
{
	this->kernel = kernel;
	kernelSet = true;
}

//This function is called by the OpenCLContext when created
void OpenCLKernel::SetContext(const OpenCLContext* const context)
{
	this->context = context;
}

} /* namespace Helper */
} /* namespace ATML */
