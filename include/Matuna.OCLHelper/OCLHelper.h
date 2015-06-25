/*
 * OCLHelper.h
 *
 *  Created on: Apr 27, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_OCLHELPER_OCLHELPER_H_
#define MATUNA_MATUNA_OCLHELPER_OCLHELPER_H_

#include "OCLInclude.h"
#include <vector>
#include <memory>

#include "OCLDeviceInfo.h"
#include "OCLPlatformInfo.h"
#include "OCLContext.h"

using namespace std;

namespace Matuna
{
namespace Helper
{

/**
 *@brief This class may be seen as a static factory class that creates objects necessary in order to create OCLDevice.
 */
class OCLHelper
final
{

	private:
		static OCLPlatformInfo GetPlatformInfo(cl_platform_id platformID);
		static OCLDeviceInfo GetDeviceInfo(
				const OCLPlatformInfo& platformInfo, cl_device_id deviceID);

	public:
		/**
		 *@brief Returns a vector that contains information about all the available platforms on the system.
		 *@return A vector of OCLPlatformInfo
		 */
		static vector<OCLPlatformInfo> GetPlatformInfos();

		/**
		 *@brief Returns a vector that contains information about all the available platforms on the system.
		 *@param platformInfo A OCLPlatformInfo
		 *@return A vector of OCLDeviceInfo
		 */
		static vector<OCLDeviceInfo> GetDeviceInfos(
				const OCLPlatformInfo& platformInfo);

		static unique_ptr<OCLContext> GetContext(
				const OCLPlatformInfo& platformInfo, int queuesPerDevice = 1,
				cl_command_queue_properties queueType = 0);

		static unique_ptr<OCLContext> GetContext(const OCLPlatformInfo& platformInfo,
				vector<tuple<OCLDeviceConfig, OCLDeviceInfo>> deviceConfigs);
	};

	} /* namespace Helper */
	} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLHELPER_OCLHelper_H_ */
