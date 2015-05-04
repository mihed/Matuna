/*
 * OpenCLKernel.h
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLKERNEL_H_
#define ATML_OPENCLHELPER_OPENCLKERNEL_H_

#include <string>
#include <vector>
#include <memory>
#include <tuple>

#include "OpenCLMemory.h"

using namespace std;

namespace ATML {
	namespace Helper {

		/**
		*@brief This class serves as an abstract base for every OpenCLKernel that are to be executed on the OpenCLDevice.
		*/
		class OpenCLKernel {

		protected:
			static string GetTextFromPath(string path);

		public:
			OpenCLKernel();
			virtual ~OpenCLKernel();

			/**
			*@brief The name of the program that has the implementation of this kernel
			*
			*Every program name is different as long as their program code is different.
			*
			*@return a string representing the program name. 
			*/
			virtual string ProgramName() const = 0;

			/**
			*@brief The actual code that has this kernel
			*
			*@return a string containing the entire program code.
			*/
			virtual string ProgramCode() const = 0;

			/**
			*@brief The name of the kernel that this OpenCLKernel represents
			*@return a string containing the kernel name.
			*/
			virtual string KernelName() const = 0;

			/**
			*@brief Gets the kernel arguments that has a OpenCL buffer (memory) associated to it.
			*
			*The first index of the tuple repesents the argument index. The second index of tuple represent the value
			*that is to be passed to the argument at the argument index. As indicated in the call. The OpenCLMemory is
			*partially owned by the kernel and should not be retained longer than necessary in order to perform the native call.
			*
			*@return a constant reference to the memory arguments.
			*@warning the reference is ONLY valid as long as the OpenCLKernel is alive.
			*/
			virtual const vector<tuple<int, shared_ptr<OpenCLMemory>>>& GetMemoryArguments() const = 0;

			/**
			*@brief Gets the kernel arguments that has an arbitrary value.
			*
			*The first index of the tuple represents the argument index. The secon index of tuple represents the size of the argument.
			*The third index of the tuple is a pointer to the memory where the argument is stored. 
			*The must be stored inside the OpenCLKernel object and is valid as long as the OpenCLKernel is valid.
			*
			*@return a constant reference to arguments.
			*@warning the reference is ONLY valid as long as the OpenCLKernel is alive.
			*/
			virtual const vector<tuple<int, size_t, void*>>& GetOtherArguments() const = 0;

			/**
			*@brief Gets the global work size of the OpenCLKernel
			*
			*Every OpenCL kernel call has a global work size that defines the dimension of the kernel call.
			*This function returns the amount of dimensions and the size in every dimension of the kernel call.
			*
			*@return A constant vector reference with the global work size dimensions.
			*@warning the reference is ONLY valid as long as the OpenCLKernel is alive.
			*/
			virtual const vector<size_t>& GlobalWorkSize() const = 0;

			/**
			*@brief Gets the local work size of the OpenCLKernel
			*
			*Every OpenCl kernel call has a global and a local work size.
			*When a kernel is executed on a device, the local work group is executed on the same compute unit
			*and may have synchronization operations inside this group.
			*
			*@return A constant vector reference with the local work size dimensions.
			*@warning the reference is ONLY valid as long as the OpenCLKernel is alive.
			*/
			virtual const vector<size_t>& LocalWorkSize() const = 0;
		};

	} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLKERNEL_H_ */
