/*
 * OpenCLMemory.cpp
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#include "OpenCLMemory.h"
#include "OpenCLUtility.h"

namespace ATML {
	namespace Helper {

		OpenCLMemory::OpenCLMemory(const cl_mem memory,
			const OpenCLDevice* const owningDevice,
			const cl_mem_flags readWriteFlag) :
			memory(memory), owningDevice(owningDevice), readWriteFlag(readWriteFlag) {

		}

		OpenCLMemory::~OpenCLMemory() {
			if (memory)
				CheckOpenCLError(clReleaseMemObject(memory),
				"Could not release the open cl memory object");
		}

	} /* namespace Helper */
} /* namespace ATML */
