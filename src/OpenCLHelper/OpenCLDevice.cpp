/*
 * OpenCLDevice.cpp
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#include "OpenCLDevice.h"

namespace ATML {
namespace Helper {

	//Initialize the object with a standard single in-order command queue
	OpenCLDevice::OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo)
	{

	}

	//Initiaize the object with a single command queue define in the properties
	OpenCLDevice::OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo, cl_command_queue_properties properties)
	{

	}


OpenCLDevice::~OpenCLDevice() {
	// TODO Auto-generated destructor stub
}

} /* namespace Helper */
} /* namespace ATML */
