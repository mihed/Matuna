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
		const OCLContext* const owningContext,
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

} /* namespace Helper */
} /* namespace Matuna */
