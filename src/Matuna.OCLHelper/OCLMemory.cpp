/*
* OCLMemory.cpp
*
*  Created on: Apr 26, 2015
*      Author: Mikael
*/

#include "OCLMemory.h"
#include "OCLUtility.h"

namespace Matuna
{
	namespace Helper
	{

		OCLMemory::OCLMemory(cl_mem memory,
			OCLContext* owningContext,
			const cl_mem_flags readWriteFlag, size_t byteSize) :
		memory(memory), owningContext(owningContext), readWriteFlag(
			readWriteFlag), byteSize(byteSize)
		{

		}

		OCLMemory::~OCLMemory()
		{
			if (memory)
				CheckOCLError(clReleaseMemObject(memory),
				"Could not release the open cl memory object");
		}

		cl_mem_flags OCLMemory::ReadWriteFlag() const
		{
			return readWriteFlag;
		}

		OCLContext* OCLMemory::OwningContext() const
		{
			return owningContext;
		}

		cl_mem OCLMemory::GetCLMemory() const
		{
			return memory;
		}

		size_t OCLMemory::ByteSize() const
		{
			return byteSize;
		}

	} /* namespace Helper */
} /* namespace Matuna */
