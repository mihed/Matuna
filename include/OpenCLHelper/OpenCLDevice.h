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
#include "OpenCLMemory.h"
#include "OpenCLDeviceConfig.h"

using namespace std;

namespace ATML
{
namespace Helper
{

class OpenCLContext;

/**
 *@brief An OpenCL device on the system.
 *
 *This device is responsible for executing kernels and creating memory on the device.
 */
class OpenCLDevice
final
{
	private:
		const OpenCLContext* const context;
		vector<cl_command_queue> queues;
		cl_device_id deviceID;
		OpenCLDeviceInfo deviceInfo;
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
		OpenCLDevice(const OpenCLContext* const context,
				const OpenCLDeviceInfo& deviceInfo,
				const vector<cl_command_queue>& queues);

		~OpenCLDevice();

		/**
		 *@brief Execute the kernel on the device
		 *
		 *@param kernel OpenCLKernel that is to be executed
		 *@param blocking True if the call should block the execution.
		 */
		void ExecuteKernel(const OpenCLKernel* kernel, int queueIndex = 0,
				bool blocking = true);

		/**
		 *@brief Waits for the execution of all kernels to finish.
		 *
		 *
		 */
		void WaitForDeviceQueue(int queueIndex = 0);

		/**
		 *@brief Writes memory from the host device to the OpenCLDevice.
		 *
		 *@param memory The OpenCLMemory to be written to.
		 *@param bytes The number of bytes to be written to the OpenCLMemory.
		 *@param buffer The buffer to be written to the OpenCLMemory.
		 *@param blockingCall True if the call is blocking.
		 */
		void WriteMemory(OpenCLMemory* memory, size_t bytes, void* buffer,
				int queueIndex = 0, bool blockingCall = true);

		/**
		 *@brief Reads the memory from the device to the host device.
		 *
		 *@param memory The OpenCLMemory containing the memory
		 *@param bytes The amount of bytes to be written to the buffer.
		 *@param buffer The buffer into which the OpenCLMemory will write its memory.
		 *@param blockingCall True if the call is blocking.
		 */
		void ReadMemory(OpenCLMemory* memory, size_t bytes, void* buffer,
				int queueIndex = 0, bool blockingCall = true);

		/**
		 *@brief Returns the OpenCLDeviceInfo describing this device.
		 *
		 *
		 *@return OpenCLDeviceInfo
		 */
		OpenCLDeviceInfo DeviceInfo() const
		{
			return deviceInfo;
		}
		;

		cl_device_id DeviceID() const
		{
			return deviceID;
		}
		;
	};

	}
	/* namespace Helper */
	} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICE_H_ */
