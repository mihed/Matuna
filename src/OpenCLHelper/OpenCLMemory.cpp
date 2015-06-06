/*
 * OpenCLMemory.cpp
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#include "OpenCLMemory.h"
#include "OpenCLUtility.h"

namespace Matuna
{
namespace Helper
{

OpenCLMemory::OpenCLMemory(cl_mem memory,
		const OpenCLContext* const owningContext,
		const cl_mem_flags readWriteFlag, size_t byteSize) :
		memory(memory), owningContext(owningContext), readWriteFlag(
				readWriteFlag), byteSize(byteSize)
{

}

OpenCLMemory::~OpenCLMemory()
{
	if (memory)
		CheckOpenCLError(clReleaseMemObject(memory),
				"Could not release the open cl memory object");
}

} /* namespace Helper */
} /* namespace Matuna */
