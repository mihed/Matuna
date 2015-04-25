#ifndef ATML_OPENCLHELPER_OPENCLMEMORY_H_
#define ATML_OPENCLHELPER_OPENCLMEMORY_H_
#include "OpenCLMemory.h"
#endif

#ifndef ATML_ATML_HELPER_OPENCLUTILITY_H_
#define ATML_ATML_HELPER_OPENCLUTILITY_H_
#include "OpenCLUtility.h"
#endif

namespace ATML
{
	namespace Helper
	{
		OpenCLMemory::OpenCLMemory(const cl_mem memory, const OpenCLDevice* const owningDevice, const cl_mem_flags readWriteFlag)
			: memory(memory), owningDevice(owningDevice), readWriteFlag(readWriteFlag)
		{

		}


		OpenCLMemory::~OpenCLMemory()
		{
			if (memory)
				CheckOpenCLError(clReleaseMemObject(memory), "Could not release the open cl memory object");
		}
	}
}
