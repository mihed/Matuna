/*
 * OpenCLContext.h
 *
 *  Created on: May 6, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLCONTEXT_H_
#define ATML_OPENCLHELPER_OPENCLCONTEXT_H_

#include <CL/cl.h>
#include <memory>
#include <tuple>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <stdexcept>

#include "OpenCLUtility.h"
#include "OpenCLDeviceConfig.h"
#include "OpenCLDeviceInfo.h"
#include "OpenCLDevice.h"
#include "OpenCLKernel.h"
#include "OpenCLKernelProgram.h"
#include "OpenCLMemory.h"

using namespace std;

namespace ATML
{
namespace Helper
{

class OpenCLContext
final
{
	private:
		cl_context context;
		vector<unique_ptr<OpenCLDevice>> devices;
		unordered_map<string, cl_program> programs;

	public:
		OpenCLContext(
				const vector<tuple<OpenCLDeviceConfig, OpenCLDeviceInfo>>& deviceConfigs);
		~OpenCLContext();

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
		 *@return a unique pointer to OpenCLMemory
		 */
		unique_ptr<OpenCLMemory> CreateMemory(cl_mem_flags flags,
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
		 *	If specified, it indicates that the application wants the OpenCL implementation to use memory referenced by host_ptr as the storage bits for the memory object.
		 *	OpenCL implementations are allowed to cache the buffer contents pointed to by host_ptr in device memory. This cached copy can be used when kernels are executed on a device.
		 *	The result of OpenCL commands that operate on multiple buffer objects created with the same host_ptr or overlapping host regions is considered to be undefined.
		 *  -CL_MEM_ALLOC_HOST_PTR: This flag specifies that the application wants the OpenCL implementation to allocate memory from host accessible memory.
		 *	CL_MEM_ALLOC_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually exclusive.
		 *  -CL_MEM_COPY_HOST_PTR: This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OpenCL implementation to allocate memory for the memory object and copy the data from memory referenced by host_ptr.
		 *	CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually exclusive.
		 *	CL_MEM_COPY_HOST_PTR can be used with CL_MEM_ALLOC_HOST_PTR to initialize the contents of the cl_mem object allocated using host-accessible (e.g. PCIe) memory.
		 *@param flags Valid flags are: CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY, CL_MEM_USE_HOST_PTR, CL_MEM_ALLOC_HOST_PTR and CL_MEM_COPY_HOST_PTR
		 *@param bytes The size of the memory chunk to be allocated.
		 *@param buffer The buffer that we be copied (or referenced depending on the flags) into the OpenCLMemory.
		 *@return a unique pointer to OpenCLMemory
		 */
		unique_ptr<OpenCLMemory> CreateMemory(cl_mem_flags flags, size_t bytes,
				void* buffer) const;

		size_t DeviceCount() const
		{
			return devices.size();
		}
		;

		vector<OpenCLDevice*> GetDevices() const;

		void AddProgramFromSource(const string& programName,
				const string& compilerOptions,
				const vector<string>& programCodeFiles,
				const vector<OpenCLDevice*>& devices);

		void AddProgramFromBinary(const string& programName,
				const size_t* lengths, const unsigned char** binaries,
				const vector<OpenCLDevice*>& devices);

		void RemoveProgram(const string& programName);

		template<class T>
		unique_ptr<T> CreateOpenCLKernelProgram(
				const vector<OpenCLDevice*>& devices)
		{
			static_assert(is_base_of<OpenCLKernelProgram, T>::value, "The type is not an OpenCLKernelProgram");
			unique_ptr<OpenCLKernelProgram> result(new T());

			if (programs.find(result->ProgramName()) == programs.end())
				throw runtime_error(
						"The program where the kernel belongs has not been added.");

			AddProgramFromSource(result->ProgramName(),
					result->GetCompilerOptions(), result->GetProgramCode(),
					devices);

			auto& program = programs[result->ProgramName()];
			cl_int errorCode;
			cl_kernel kernelToSet = clCreateKernel(program,
					result->KernelName().c_str(), &errorCode);
			CheckOpenCLError(errorCode, "Could not create the kernel");
			result->SetOCLKernel(kernelToSet);

			if (!result->KernelSet())
				throw runtime_error("The kernel could not be set");

			return move(result);
		}

		template<class T>
		unique_ptr<T> CreateOpenCLKernel()
		{
			static_assert(is_base_of<OpenCLKernel, T>::value, "The type is not an OpenCLKernel");
			unique_ptr<OpenCLKernel> result(new T());

			if (programs.find(result->ProgramName()) == programs.end())
				throw runtime_error(
						"The program where the kernel belongs has not been added.");

			auto& program = programs[result->ProgramName()];
			cl_int errorCode;
			cl_kernel kernelToSet = clCreateKernel(program,
					result->KernelName().c_str(), &errorCode);
			CheckOpenCLError(errorCode, "Could not create the kernel");
			result->SetOCLKernel(kernelToSet);

			if (!result->KernelSet())
				throw runtime_error("The kernel could not be set");

			return move(result);
		}
		;
	};

	} /* namespace Helper */
	} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLCONTEXT_H_ */
