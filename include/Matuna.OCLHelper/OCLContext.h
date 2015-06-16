/*
* OCLContext.h
*
*  Created on: May 6, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLHELPER_OCLCONTEXT_H_
#define MATUNA_OCLHELPER_OCLCONTEXT_H_

#include "OCLInclude.h"
#include <memory>
#include <tuple>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <stdexcept>

#include "OCLUtility.h"
#include "OCLProgram.h"
#include "OCLDeviceConfig.h"
#include "OCLDeviceInfo.h"
#include "OCLDevice.h"
#include "OCLKernel.h"
#include "OCLMemory.h"

using namespace std;

namespace Matuna
{
	namespace Helper
	{

		class OCLContext
			final
		{
		private:
			cl_context context;
			vector<unique_ptr<OCLDevice>> devices;
			unordered_map<string, tuple<unique_ptr<OCLProgram>, unordered_set<OCLDevice*>>> programs;
			OCLPlatformInfo platformInfo;

		public:
			OCLContext(const OCLPlatformInfo& platformInfo,
				const vector<tuple<OCLDeviceConfig, OCLDeviceInfo>>& deviceConfigs);
			~OCLContext();

			/**
			*@brief Creates an empty memory on the device.
			*
			*The memory object follows RAII. When the memory is deleted, the resources on the device are released.
			*
			*  -CL_MEM_READ_WRITE:	This flag specifies that the memory object will be read and written by a kernel. This is the default.
			*  -CL_MEM_WRITE_ONLY: This flags specifies that the memory object will be written but not read by a kernel. Reading from a buffer or image object created with CL_MEM_WRITE_ONLY inside a kernel is undefined.
			*  -CL_MEM_READ_ONLY: This flag specifies that the memory object is a read-only memory object when used inside a kernel.
			*	Writing to a buffer or image object created with CL_MEM_READ_ONLY inside a kernel is undefined.
			*
			*@param flags Valid flags are: CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY and CL_MEM_READ_ONLY
			*@param bytes The size of the memory chunk to be allocated.
			*@return a unique pointer to OCLMemory
			*/
			unique_ptr<OCLMemory> CreateMemory(cl_mem_flags flags,
				size_t bytes) const;

			/**
			*@brief
			*
			*The memory object follows RAII. When the memory is deleted, the resources on the device are released.
			*
			*
			*  -CL_MEM_READ_WRITE:	This flag specifies that the memory object will be read and written by a kernel. This is the default.
			*  -CL_MEM_WRITE_ONLY: This flags specifies that the memory object will be written but not read by a kernel.
			*	Reading from a buffer or image object created with CL_MEM_WRITE_ONLY inside a kernel is undefined.
			*  -CL_MEM_READ_ONLY: This flag specifies that the memory object is a read-only memory object when used inside a kernel.
			*	Writing to a buffer or image object created with CL_MEM_READ_ONLY inside a kernel is undefined.
			*  -CL_MEM_USE_HOST_PTR: This flag is valid only if host_ptr is not NULL.
			*	If specified, it indicates that the application wants the OCL implementation to use memory referenced by host_ptr as the storage bits for the memory object.
			*	OCL implementations are allowed to cache the buffer contents pointed to by host_ptr in device memory. This cached copy can be used when kernels are executed on a device.
			*	The result of OCL commands that operate on multiple buffer objects created with the same host_ptr or overlapping host regions is considered to be undefined.
			*  -CL_MEM_ALLOC_HOST_PTR: This flag specifies that the application wants the OCL implementation to allocate memory from host accessible memory.
			*	CL_MEM_ALLOC_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually exclusive.
			*  -CL_MEM_COPY_HOST_PTR: This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OCL implementation to allocate memory for the memory object and copy the data from memory referenced by host_ptr.
			*	CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually exclusive.
			*	CL_MEM_COPY_HOST_PTR can be used with CL_MEM_ALLOC_HOST_PTR to initialize the contents of the cl_mem object allocated using host-accessible (e.g. PCIe) memory.
			*@param flags Valid flags are: CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY, CL_MEM_USE_HOST_PTR, CL_MEM_ALLOC_HOST_PTR and CL_MEM_COPY_HOST_PTR
			*@param bytes The size of the memory chunk to be allocated.
			*@param buffer The buffer that we be copied (or referenced depending on the flags) into the OCLMemory.
			*@return a unique pointer to OCLMemory
			*/
			unique_ptr<OCLMemory> CreateMemory(cl_mem_flags flags, size_t bytes,
				void* buffer) const;

			size_t DeviceCount() const
			{
				return devices.size();
			}
			;

			void AttachProgram(unique_ptr<OCLProgram> program, const vector<OCLDevice*>& affectedDevicesk);
			void DetachProgram(OCLProgram* program);
			OCLProgram* GetProgram(string name) const;

			bool ProgramAdded(const string& name) const;

			OCLPlatformInfo GetPlatformInfo() const;

			vector<OCLDevice*> GetDevices() const;

			//Seems like the binary is buggy
			/*
			void AddProgramFromBinary(const string& programName,
			const vector<vector<unsigned char>>& binaries,
			const vector<OCLDevice*>& affectedDevices);

			vector<vector<unsigned char>> GetBinaryProgram(const string& programName);
			*/
		};

	} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_OCLHELPER_OCLCONTEXT_H_ */
