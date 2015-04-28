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

namespace ATML {
namespace Helper {

class OpenCLDevice final{
private:
	cl_context context;
	cl_command_queue queue;
	cl_device_id deviceID;
	OpenCLDeviceInfo deviceInfo;
	unordered_map<string, tuple<int, unordered_map<string, int>>> referenceCounter;
	unordered_map<string, tuple<cl_program, unordered_map<string, cl_kernel>>> programsAndKernels;

public:
	//Initialize the object with a standard single in-order command queue
	OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo);

	//Initiaize the object with a single command queue define in the properties
	OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo, cl_command_queue_properties properties);

	~OpenCLDevice();

	void AddKernel(const OpenCLKernel* kernel);
	void RemoveKernel(const OpenCLKernel* kernel);
	void ExecuteKernel(const OpenCLKernel* kernel, bool blocking = true);
	void WaitForDeviceQueue();

	//Creates empty memory on this device with the given size
	unique_ptr<OpenCLMemory> CreateMemory(cl_mem_flags flags, size_t bytes);

	//Creates memory on this device with the values defined in the buffer.
	//The size of the buffer must be the same as the created memory
	unique_ptr<OpenCLMemory> CreateMemory(cl_mem_flags flags, size_t bytes, void* buffer);

private:
	bool ProgramAdded(string programName);
	bool KernelAdded(string programName, string kernelName);
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICE_H_ */
