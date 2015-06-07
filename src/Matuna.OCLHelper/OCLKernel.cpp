/*
 * OCLKernel.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */

#include "OCLKernel.h"
#include <iostream>
#include <sstream>
#include <fstream>

namespace Matuna
{
namespace Helper
{

int OCLKernel::instanceCounter = 0;

OCLKernel::OCLKernel()
{
	instanceCounter++;
	kernel = nullptr;
	context = nullptr;
	kernelSet = false;
}

OCLKernel::~OCLKernel()
{

}

void OCLKernel::SetOCLKernel(cl_kernel kernel)
{
	this->kernel = kernel;
	kernelSet = true;
}

//This function is called by the OCLContext when created
void OCLKernel::SetContext(const OCLContext* const context)
{
	this->context = context;
}

} /* namespace Helper */
} /* namespace Matuna */
