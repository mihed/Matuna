/*
* OCLKernel.h
*
*  Created on: Apr 28, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLHELPER_OCLKERNEL_H_
#define MATUNA_OCLHELPER_OCLKERNEL_H_

#include "OCLInclude.h"
#include <string>
#include <vector>

using namespace std;

namespace Matuna
{
	namespace Helper
	{

		class OCLProgram;

		/**
		*@brief This class serves as an abstract base for every OCLKernel that are to be executed on the OCLDevice.
		*/
		class OCLKernel
		{
			friend class OCLProgram;

		private:
			const OCLProgram* owningProgram;
			cl_kernel kernel;

		private:
			//This function is called by the OCL program when attached to it.
			void SetKernel(cl_kernel kernel);
			void SetProgram(const OCLProgram* const program);

		public:
			OCLKernel();
			virtual ~OCLKernel();

			bool KernelSet() const;
			bool ProgramSet() const;
			cl_kernel GetKernel() const;
			const OCLProgram* const GetProgram() const;

			/**
			*@brief The name of the kernel that this OCLKernel represents
			*@return a string containing the kernel name.
			*/
			virtual string Name() const = 0;

			/**
			*@brief Gets the global work size of the OCLKernel
			*
			*Every OCL kernel call has a global work size that defines the dimension of the kernel call.
			*This function returns the amount of dimensions and the size in every dimension of the kernel call.
			*
			*@return A constant vector reference with the global work size dimensions.
			*@warning the reference is ONLY valid as long as the OCLKernel is alive.
			*/
			virtual const vector<size_t>& GlobalWorkSize() const = 0;

			/**
			*@brief Gets the local work size of the OCLKernel
			*
			*Every OpenCl kernel call has a global and a local work size.
			*When a kernel is executed on a device, the local work group is executed on the same compute unit
			*and may have synchronization operations inside this group.
			*
			*@return A constant vector reference with the local work size dimensions.
			*@warning the reference is ONLY valid as long as the OCLKernel is alive.
			*/
			virtual const vector<size_t>& LocalWorkSize() const = 0;
		};

	}
	/* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_OCLHELPER_OCLKERNEL_H_ */
