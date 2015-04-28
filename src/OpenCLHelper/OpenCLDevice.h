/*
 * OpenCLDevice.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICE_H_
#define ATML_OPENCLHELPER_OPENCLDEVICE_H_

#include "OpenCLDeviceInfo.h"
#include "OpenCLKernel.h"
#include <CL/cl.h>
#include <unordered_map>
#include <tuple>

using namespace std;

namespace ATML {
namespace Helper {

class OpenCLDevice final{
private:
	cl_context context;
	cl_command_queue queue;
	unordered_map<string, tuple<int, unordered_map<string, int>>> referenceCounter;
public:
	//Initialize the object with a standard single in-order command queue
	OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo);

	//Initiaize the object with a single command queue define in the properties
	OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo, cl_command_queue_properties properties);

	~OpenCLDevice();

	void AddKernel(const OpenCLKernel& kernel);
	void RemoveKernel(const OpenCLKernel& kernel);
	void ExecuteKernel(const OpenCLKernel& kernel, bool blocking = true);
	void WaitForDeviceQueue();
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICE_H_ */
