/*
 * OCLPlatformInfo.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_OCLHELPER_OCLPLATFORMINFO_H_
#define MATUNA_MATUNA_OCLHELPER_OCLPLATFORMINFO_H_

#include <string>
#include "OCLInclude.h"

using namespace std;

namespace Matuna {
	namespace Helper {

		/**
		*@brief A wrapper of the native platform information. See OCLHelper to obtain it.
		*/
		class OCLPlatformInfo final{

		private:
			cl_platform_id platformID;

			string platformName;
			string platformProfile;
			string platformVersion;
			string platformVendor;
			string platformExtensions;

		public:
			/**
			*@param platformInfo The native id to the platform. Obtained by the corresponding native call.
			*@param platformName The human readable name of the platform. Obtained by the corresponding native call.
			*@param platformProfile The platform profile. Obtained by the corresponding native call.
			*@param platformVersion The OCL version of the platform. Obtained by the corresponding native call.
			*@param platformExtensions The extensions supported by the OCL platform.
			*/
			OCLPlatformInfo(const cl_platform_id platformID,
				const string& platformName, const string& platformProfile,
				const string& platformVersion, const string& platformVendor,
				const string& platformExtensions);

			~OCLPlatformInfo();

			/**
			*@breif Gets a formatted string containing all the information about this platform.
			*@return The string containing all the platform information.
			*/
			string GetString() const;

			/**
			*@brief The native OCL id of the platform.
			*
			* This id is used by the OCLHelper in order to fetch corresponding OCLDevice that resides on this platform.
			*
			*@return The native OCL id of the platform
			*/
			cl_platform_id PlatformID() const
			{
				return platformID;
			}
			;
			/**
			*@brief The name of the OCL platform.
			*
			* On a given system, one may have many different OCL platforms. Use this function
			* together with the other functions in this class to target a specific OCL platform.
			*
			*@return Human readable string of the platform name .
			*/
			string PlatformName() const {
				return platformName;
			}
			;
			/**
			*@brief Every OCL platform can have either FULL_PROFILE or EMBEDDED_PROFILE
			*
			* FULL_PROFILE supports the entire OCL specification profile, depending on the version.
			* We target the OCL 1.1 profile here since we want it to run on Nvidia cards as well.
			*
			*
			* EMBEDDED_PROFILE if the OCL implementation supports the specifications for the embedded profile.
			* the embedded profile is a subset of the full profile. For example:
			* - 64-bit integer support is optional.
			* - Support for 3D images is optional.
			* - Support for 2D image array writes is optional
			*
			* @return FULL_PROFILE or EMBEDDED_PROFILE
			*/
			string PlatformProfile() const {
				return platformProfile;
			}
			;

			/**
			*@brief The OCL version string
			*@return OCL<space><major_version.minor_version><space><platform-specific information>
			*/
			string PlatformVersion() const {
				return platformVersion;
			}
			;
			/**
			*@brief Returns all the extensions supported by this platform. All the devices on the platform must support these extensions.
			*@return A space-separated list of extension names.
			*/
			string PlatformExtensions() const {
				return platformExtensions;
			}
			;

			/**
			*@brief Platform vendor string
			*@return The platform vendor string
			*/
			string PlatformVendor() const {
				return platformVendor;
			}
			;
		};

	} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLHELPER_OCLPLATFORMINFO_H_ */
