/*
 * OpenCLDevice.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICE_H_
#define ATML_OPENCLHELPER_OPENCLDEVICE_H_

#include "OpenCLDeviceInfo.h"
#include <CL/cl.h>

namespace ATML {
namespace Helper {

class OpenCLDevice {
public:
	//Initialize the object with a standard single in-order command queue
	OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo);

	//Initiaize the object with a single command queue define in the properties
	OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo, cl_command_queue_properties properties);

	~OpenCLDevice();
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICE_H_ */
