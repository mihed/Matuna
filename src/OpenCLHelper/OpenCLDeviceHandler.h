/*
 * OpenCLDeviceHandler.h
 *
 *  Created on: Apr 27, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICEHANDLER_H_
#define ATML_OPENCLHELPER_OPENCLDEVICEHANDLER_H_

#include <CL/cl.h>
#include <vector>
#include <memory>

#include "OpenCLDeviceInfo.h"
#include "OpenCLPlatformInfo.h"
#include "OpenCLDevice.h"

using namespace std;

namespace ATML {
	namespace Helper {

		/**
		*@brief This class may be seen as a static factory class that creates objects necessary in order to create OpenCLDevice.
		*/
		class OpenCLDeviceHandler final{

		private:
			static OpenCLPlatformInfo GetPlatformInfo(cl_platform_id platformID);
			static OpenCLDeviceInfo GetDeviceInfo(const OpenCLPlatformInfo& platformInfo,
				cl_device_id deviceID);

		public:
			/**
			*@brief Returns a vector that contains information about all the available platforms on the system.
			*@return A vector of OpenCLPlatformInfo
			*/
			static vector<OpenCLPlatformInfo> GetPlatformInfos();

			/**
			*@brief Returns a vector that contains information about all the available devices on the system.
			*@return A vector of OpenCLDeviceInfo
			*/
			static vector<OpenCLDeviceInfo> GetDeviceInfos();

			/**
			*@brief Returns a vector that contains information about all the available platforms on the system.
			*@param platformInfo A OpenCLPlatformInfo
			*@return A vector of OpenCLDeviceInfo
			*/
			static vector<OpenCLDeviceInfo> GetDeviceInfos(
				const OpenCLPlatformInfo& platformInfo);

			/**
			*@brief Returns a vector that contains information about the devices given in the platforms.
			*@param platformInfos A vector of OpenCLPlatformInfo
			*@return A vector of OpenCLDeviceInfo
			*/
			static vector<OpenCLDeviceInfo> GetDeviceInfos(
				const vector<OpenCLPlatformInfo>& platformInfos);

			/**
			*@brief Returns a vector of OpenCLDevice that are found in the given OpenCLPlatformInfo.
			*
			*The returned devices follows the RAII pattern. When they are deleted so are all their resources.
			*
			*@param platformInfo OpenCLPlatformInfo
			*@return A vector of OpenCLDevice
			*/
			static vector<unique_ptr<OpenCLDevice>> GetDevices(
				const OpenCLPlatformInfo& platformInfo);

			/**
			*@brief Returns a vector of OpenCLDevice that are found in the vector of OpenCLPlatformInfo.
			*
			*The returned devices follows the RAII pattern. When they are deleted so are all their resources.
			*
			*@param A vector of OpenCLPlatformInfo 
			*@return A vector of OpenCLDevice
			*/
			static vector<unique_ptr<OpenCLDevice>> GetDevices(
				const vector<OpenCLPlatformInfo>& platformInfos);

			/**
			*@brief Returns a vector of all OpenCLDevice that are found on the system.
			*
			*The returned devices follows the RAII pattern. When they are deleted so are all their resources.
			*
			*@return A vector of OpenCLDevice
			*/
			static vector<unique_ptr<OpenCLDevice>> GetDevices();

			/**
			*@brief Returns a vector of all OpenCLDevice that are given in the vector of OpenCLDeviceInfo
			*
			*The returned devices follows the RAII pattern. When they are deleted so are all their resources.
			*
			*@param A vector of OpenCLDeviceInfo
			*@return A vector of OpenCLDevice
			*/
			static vector<unique_ptr<OpenCLDevice>> GetDevices(
				const vector<OpenCLDeviceInfo>& deviceInfos);

			/**
			*@brief Returns a single OpenCLDevice from the given OpenCLDeviceInfo
			*
			*The returned device follows the RAII pattern. When it's deleted so are all its resources.
			*
			*@param A OpenCLDeviceInfo
			*@return A OpenCLDevice
			*/
			static unique_ptr<OpenCLDevice> GetDevices(
				const OpenCLDeviceInfo& deviceInfo);
		};

	} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICEHANDLER_H_ */
