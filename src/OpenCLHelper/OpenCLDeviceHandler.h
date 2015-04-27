/*
 * OpenCLDeviceHandler.h
 *
 *  Created on: Apr 27, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICEHANDLER_H_
#define ATML_OPENCLHELPER_OPENCLDEVICEHANDLER_H_

#include <CL\cl.h>
#include <vector>
#include <memory>

#include "OpenCLDeviceInfo.h"
#include "OpenCLPlatformInfo.h"
#include "OpenCLDevice.h"

using namespace std;

namespace ATML {
namespace Helper {

class OpenCLDeviceHandler {

private:
	OpenCLPlatformInfo GetPlatformInfo(cl_platform_id platformID);
	OpenCLDeviceInfo GetDeviceInfo(const OpenCLPlatformInfo& platformInfo,
			cl_device_id deviceID);

public:
	OpenCLDeviceHandler();
	~OpenCLDeviceHandler();

	//Returns information about all the currently installed OpenCL platforms.
	vector<OpenCLPlatformInfo> GetPlatformInfos();

	//Returns information about all OpenCL devices inside all the OpenCL platforms
	vector<OpenCLDeviceInfo> GetDeviceInfos();

	//Returns information about all OpenCL devices present inside the platforms given
	//as input argument.
	vector<OpenCLDeviceInfo> GetDeviceInfos(
			const OpenCLPlatformInfo& platformInfos);

	//Returns information about all OpenCL devices present inside the platforms given
	//as input argument.
	vector<OpenCLDeviceInfo> GetDeviceInfos(
			const vector<OpenCLPlatformInfo>& platformInfos);

	//Returns all the devices given in the specified platform information
	vector<shared_ptr<OpenCLDevice>> GetDevices(
			const OpenCLPlatformInfo& platformInfo);

	//Returns all the devices given in the specified platform information
	vector<shared_ptr<OpenCLDevice>> GetDevices(
			const vector<OpenCLPlatformInfo>& platformInfos);

	//Returns all of the present devices
	vector<shared_ptr<OpenCLDevice>> GetDevices();

	//Returns the devices given by the device info
	vector<shared_ptr<OpenCLDevice>> GetDevices(
			const vector<OpenCLDeviceInfo>& deviceInfos);
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICEHANDLER_H_ */
