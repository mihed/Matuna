/*
* OCLKernel.cpp
*
*  Created on: Apr 28, 2015
*      Author: Mikael
*/

#include "OCLKernel.h"
#include "OCLUtility.h"
#include <iostream>
#include <sstream>
#include <fstream>

namespace Matuna
{
	namespace Helper
	{


		OCLKernel::OCLKernel()
		{
			kernel = nullptr;
			owningProgram = nullptr;
		}

		OCLKernel::~OCLKernel()
		{
			if (KernelSet())
				CheckOCLError(clReleaseKernel(kernel), "The kernel could not be released");
		}

		void OCLKernel::ProgramDetach()
		{
			if (!ProgramSet())
				throw runtime_error("The program cannot detach itself if it has not been attached");

			if (KernelSet())
				CheckOCLError(clReleaseKernel(kernel), "The kernel could not be released");

			kernel = nullptr;
			owningProgram = nullptr;
		}

		bool OCLKernel::KernelSet() const
		{
			return kernel == nullptr ? false : true;
		}

		bool OCLKernel::ProgramSet() const
		{
			return owningProgram == nullptr ? false : true;
		}

		cl_kernel OCLKernel::GetKernel() const
		{
			return kernel;
		}

		void OCLKernel::SetKernel(cl_kernel kernel)
		{
			if (kernel == nullptr)
				throw invalid_argument("Null pointer argument");

			this->kernel = kernel;
		}

		//This function is called by the OCLContext when created
		void OCLKernel::SetProgram(const OCLProgram* const program)
		{
			if (program == nullptr)
				throw invalid_argument("Null pointer argument");

			this->owningProgram = program;
		}

		const OCLProgram* const OCLKernel::GetProgram() const
		{
			return this->owningProgram;
		}

	} /* namespace Helper */
} /* namespace Matuna */
