/*
 * OpenCLMemory.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef OPENCLHELPER_OPENCLMEMORY_H_
#define OPENCLHELPER_OPENCLMEMORY_H_

#include <CL/cl.h>

namespace ATML {
	namespace Helper {

		class OpenCLDevice;

		/**
		*@brief This class wrapps a native OpenCL memory pointer. Follows the RAII pattern, meaning that when the destructor is called the OCL memory is released.
		*
		*In this version of the library. The memory object is bound to the OpenCLDevice.
		*It must be used in conjunction to the OpenCLDevice that created the memory object.
		*In future version we can expect memory objects to be bound to the platform (context) instead.
		*/
		class OpenCLMemory final
		{
			friend class OpenCLDevice;

		private:
			const cl_mem memory;
			const cl_mem_flags readWriteFlag;

			//TODO: In the future, this should be replaced by the context since we can, in some cases, share memory between devices
			const OpenCLDevice* const owningDevice;
		public:

			/**
			*@brief Instantiate the OpenCLMemory. Normally called inside OpenCLDevice::CreateMemory
			*
			*The normal usage of this function is throug the OpenCLDevice.
			*/
			OpenCLMemory(cl_mem memory, const OpenCLDevice* const owningDevice, cl_mem_flags readWriteFlag);
			~OpenCLMemory();

			/**
			*@brief The accessor of the memory: read, write or read-write.
			*@return The OpenCL memory accessor flag of this memor.
			*/
			cl_mem_flags ReadWriteFlag() const { return readWriteFlag; };
		};

	} /* namespace Helper */
} /* namespace ATML */

#endif /* OPENCLHELPER_OPENCLMEMORY_H_ */
