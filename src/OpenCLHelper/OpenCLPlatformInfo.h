/*
 * OpenCLPlatformInfo.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLPLATFORMINFO_H_
#define ATML_OPENCLHELPER_OPENCLPLATFORMINFO_H_

#include <string>
#include <CL/cl.h>

using namespace std;

namespace ATML {
namespace Helper {

class OpenCLPlatformInfo final{

private:
	cl_platform_id platformInfo;

	string platformName;
	string platformProfile;
	string platformVersion;
	string platformVendor;
	string platformExtensions;

public:
	OpenCLPlatformInfo(const cl_platform_id platformInfo,
			const string& platformName, const string& platformProfile,
			const string& platformVersion, const string& platformVendor,
			const string& platformExtensions);

	~OpenCLPlatformInfo();

	string GetString() const;

	cl_platform_id PlatformInfo() const
	{
		return platformInfo;
	}
	;
	string PlatformName() const {
		return platformName;
	}
	;
	string PlatformProfile() const {
		return platformProfile;
	}
	;
	string PlatformVersion() const {
		return platformVersion;
	}
	;
	string PlatformExtensions() const {
		return platformExtensions;
	}
	;
	string PlatformVendor() const {
		return platformVendor;
	}
	;
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLPLATFORMINFO_H_ */
