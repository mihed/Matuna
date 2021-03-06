/*
 * OCLPlatformInfo.cpp
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#include "OCLPlatformInfo.h"
#include <sstream>

namespace Matuna {
	namespace Helper {

		OCLPlatformInfo::OCLPlatformInfo(const cl_platform_id platformID,
			const string& platformName, const string& platformProfile,
			const string& platformVersion, const string& platformVendor,
			const string& platformExtensions) :
				platformID(platformID), platformName(platformName), platformProfile(
			platformProfile), platformVersion(platformVersion), platformVendor(
			platformVendor), platformExtensions(platformExtensions) {

		}

		OCLPlatformInfo::~OCLPlatformInfo() {
		}

		string OCLPlatformInfo::GetString() const{
			stringstream stringStream;
			stringStream << "CL_PLATFORM_NAME: \t" << platformName << "\n";
			stringStream << "CL_PLATFORM_PROFILE: \t" << platformProfile << "\n";
			stringStream << "CL_PLATFORM_VERSION: \t" << platformVersion << "\n";
			stringStream << "CL_PLATFORM_VENDOR: \t" << platformVendor << "\n";
			stringStream << "CL_PLATFORM_EXTENSIONS: \t" << platformExtensions << "\n";

			return stringStream.str();
		}

	} /* namespace Helper */
} /* namespace Matuna */
