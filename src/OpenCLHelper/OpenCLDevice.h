/*
 * OpenCLDevice.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICE_H_
#define ATML_OPENCLHELPER_OPENCLDEVICE_H_

#include <CL/cl.h>
#include <unordered_map>
#include <tuple>
#include <string>
#include <memory>

#include "OpenCLDeviceInfo.h"
#include "OpenCLKernel.h"

using namespace std;

namespace ATML
{
namespace Helper
{

/**
 *@brief An OpenCL device on the system.
 *
 *This device is responsible for executing kernels and creating memory on the device.
 */
class OpenCLDevice
final
{
	private:
		cl_context context;
		cl_command_queue queue;
		cl_device_id deviceID;
		OpenCLDeviceInfo deviceInfo;
		unordered_map<string, tuple<int, unordered_map<string, int>>> referenceCounter;
		unordered_map<string, tuple<cl_program, unordered_map<string, cl_kernel>>> programsAndKernels;

	public:
		/**
		 *@brief Instantiate the device from the given context and OpenCLDeviceInfo.
		 *
		 *An OpenCL context is created with one or more devices.
		 *Contexts are used by the OpenCL runtime for managing objects such as command-queues, memory, program and kernel objects and for executing kernels on one or more devices specified in the context.
		 *The default command queue is in order execution.
		 *
		 *@param context The context on which the device runs
		 *@param deviceInfo The OpenCLDeviceInfo containing the information about the device to be created.
		 */
		OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo);

		/**
		 *@brief Instantiate the device from the given context, OpenCLDeviceInfo and the command queue properties.
		 *
		 *An OpenCL context is created with one or more devices.
		 *Contexts are used by the OpenCL runtime for managing objects such as command-queues, memory, program and kernel objects and for executing kernels on one or more devices specified in the context.
		 *The default command queue is in order execution.
		 *
		 *@param context context The context on which the device runs
		 *@param deviceInfo deviceInfo The OpenCLDeviceInfo containing the information about the device to be created.
		 *@param properties CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE or CL_QUEUE_PROFILING_ENABLE
		 */
		OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo, cl_command_queue_properties properties);

		~OpenCLDevice();

		/**
		 *@brief Add a OpenCLKernel so that it can be executed on the device.
		 *
		 *This class holds no reference to the OpenCLKernel it's only used in order to prepare the device
		 *for execution on this particular kernel.
		 *If a kernel has not been added before, it's program is compiled and sent to the device.
		 *If you add the same kernel multiple times a reference is increased for this kernel.
		 *
		 *@param kernel OpenCLKernel that is to be executed on the device
		 */
		void AddKernel(const OpenCLKernel* kernel);

		/**
		 *@brief Remove a OpenCLKernel so that it no longer can be executed on the device
		 *
		 *Releases all necessary resources that the kernel was occupying.
		 *If the same kernel has been added multiple times, it has to be removed multiple times
		 *for the reference counter to become zero.
		 *
		 *@param kernel OpenCLKernel that is no longer to be executed on the device
		 */
		void RemoveKernel(const OpenCLKernel* kernel);

		/**
		 *@brief Execute the kernel on the device
		 *
		 *@param kernel OpenCLKernel that is to be executed
		 *@param blocking True if the call should block the execution.
		 */
		void ExecuteKernel(const OpenCLKernel* kernel, bool blocking = true);

		/**
		 *@brief Waits for the execution of all kernels to finish.
		 *
		 *
		 */
		void WaitForDeviceQueue();

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
		unique_ptr<OpenCLMemory> CreateMemory(cl_mem_flags flags, size_t bytes);

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
		unique_ptr<OpenCLMemory> CreateMemory(cl_mem_flags flags, size_t bytes, void* buffer);

		/**
		 *@brief Writes memory from the host device to the OpenCLDevice.
		 *
		 *@param memory The OpenCLMemory to be written to.
		 *@param bytes The number of bytes to be written to the OpenCLMemory.
		 *@param buffer The buffer to be written to the OpenCLMemory.
		 *@param blockingCall True if the call is blocking.
		 */
		void WriteMemory(OpenCLMemory* memory, size_t bytes, void* buffer, bool blockingCall = true);

		/**
		 *@brief Reads the memory from the device to the host device.
		 *
		 *@param memory The OpenCLMemory containing the memory
		 *@param bytes The amount of bytes to be written to the buffer.
		 *@param buffer The buffer into which the OpenCLMemory will write its memory.
		 *@param blockingCall True if the call is blocking.
		 */
		void ReadMemory(OpenCLMemory* memory, size_t bytes, void* buffer, bool blockingCall = true);

		/**
		 *@brief Returns the OpenCLDeviceInfo describing this device.
		 *
		 *
		 *@return OpenCLDeviceInfo
		 */
		OpenCLDeviceInfo DeviceInfo() const
		{	return deviceInfo;};

	private:
		bool ProgramAdded(const string& programName);
		bool KernelAdded(const string& programName, const string& kernelName);
	};

}
/* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICE_H_ */
