/*
 * OpenCLMemory.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef OPENCLHELPER_OPENCLMEMORY_H_
#define OPENCLHELPER_OPENCLMEMORY_H_

#include <CL/cl.h>

namespace ATML
{
namespace Helper
{

class OpenCLContext;

/**
 *@brief This class wrapps a native OpenCL memory pointer. Follows the RAII pattern, meaning that when the destructor is called the OCL memory is released.
 *
 *In this version of the library. The memory object is bound to the OpenCLDevice.
 *It must be used in conjunction to the OpenCLDevice that created the memory object.
 *In future version we can expect memory objects to be bound to the platform (context) instead.
 */
class OpenCLMemory
final
{
	private:
		cl_mem memory;
		cl_mem_flags readWriteFlag;
		const OpenCLContext* const owningContext;
		size_t byteSize;
	public:

		/**
		 *@brief Instantiate the OpenCLMemory. Normally called inside OpenCLDevice::CreateMemory
		 *
		 *The normal usage of this function is throug the OpenCLDevice.
		 */
		OpenCLMemory(cl_mem memory, const OpenCLContext* const owningDevice,
				cl_mem_flags readWriteFlag, size_t byteSize);
		~OpenCLMemory();

		/**
		 *@brief The accessor of the memory: read, write or read-write.
		 *@return The OpenCL memory accessor flag of this memor.
		 */
		cl_mem_flags ReadWriteFlag() const
		{
			return readWriteFlag;
		}
		;

		const OpenCLContext* const OwningContext() const
		{
			return owningContext;
		}
		;

		cl_mem GetCLMemory() const
		{
			return memory;
		}
		;

		size_t ByteSize() const
		{
			return byteSize;
		}
		;
	};

	} /* namespace Helper */
	} /* namespace ATML */

#endif /* OPENCLHELPER_OPENCLMEMORY_H_ */
