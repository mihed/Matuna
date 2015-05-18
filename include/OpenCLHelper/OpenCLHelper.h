/*
 * OpenCLHelper.h
 *
 *  Created on: Apr 27, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLHELPER_H_
#define ATML_OPENCLHELPER_OPENCLHELPER_H_

#include "OpenCLInclude.h"
#include <vector>
#include <memory>

#include "OpenCLDeviceInfo.h"
#include "OpenCLPlatformInfo.h"
#include "OpenCLContext.h"

using namespace std;

namespace ATML
{
namespace Helper
{

/**
 *@brief This class may be seen as a static factory class that creates objects necessary in order to create OpenCLDevice.
 */
class OpenCLHelper
final
{

	private:
		static OpenCLPlatformInfo GetPlatformInfo(cl_platform_id platformID);
		static OpenCLDeviceInfo GetDeviceInfo(
				const OpenCLPlatformInfo& platformInfo, cl_device_id deviceID);

	public:
		/**
		 *@brief Returns a vector that contains information about all the available platforms on the system.
		 *@return A vector of OpenCLPlatformInfo
		 */
		static vector<OpenCLPlatformInfo> GetPlatformInfos();

		/**
		 *@brief Returns a vector that contains information about all the available platforms on the system.
		 *@param platformInfo A OpenCLPlatformInfo
		 *@return A vector of OpenCLDeviceInfo
		 */
		static vector<OpenCLDeviceInfo> GetDeviceInfos(
				const OpenCLPlatformInfo& platformInfo);

		static unique_ptr<OpenCLContext> GetContext(
				const OpenCLPlatformInfo& platformInfo, int queuesPerDevice = 1,
				cl_command_queue_properties queueType = 0);

		static unique_ptr<OpenCLContext> GetContext(const OpenCLPlatformInfo& platformInfo,
				vector<tuple<OpenCLDeviceConfig, OpenCLDeviceInfo>> deviceConfigs);
	};

	} /* namespace Helper */
	} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OpenCLHelper_H_ */
