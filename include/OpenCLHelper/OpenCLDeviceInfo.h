/*
 * OpenCLDeviceInfo.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICEINFO_H_
#define ATML_OPENCLHELPER_OPENCLDEVICEINFO_H_

#include <vector>
#include <CL/cl.h>
#include "OpenCLPlatformInfo.h"

namespace ATML {
namespace Helper {

class OpenCLDeviceInfo {

private:
	OpenCLPlatformInfo platformInfo;
	cl_device_id deviceID;
	cl_device_type type;
	cl_device_mem_cache_type globalMemoryCacheType;
	cl_device_local_mem_type localMemoryType;

	unsigned int vendorID;
	unsigned int maxComputeUnits;
	unsigned int maxWorkItemDimensions;
	unsigned int maxClockFrequency;
	vector<size_t> maxWorkItemSizes;
	size_t maxWorkGroupSize;
	unsigned int preferredCharVectorWidth;
	unsigned int preferredShortVectorWidth;
	unsigned int preferredIntVectorWidth;
	unsigned int preferredLongVectorWidth;
	unsigned int preferredFloatVectorWidth;
	unsigned int preferredDoubleVectorWidth;
	unsigned int preferredHalfVectorWidth;
	unsigned int nativeCharVectorWidth;
	unsigned int nativeIntVectorWidth;
	unsigned int nativeLongVectorWidth;
	unsigned int nativeFloatVectorWidth;
	unsigned int nativeDoubleVectorWidth;
	unsigned int nativeHalfVectorWidth;
	unsigned long maxMemoryAllocationSize;
	bool imageSupport;
	size_t maxParametersSize;
	unsigned long globalMemoryCacheSize;
	unsigned long globalMemorySize;
	unsigned long maxConstantBufferSize;
	unsigned int maxConstantArguments;
	unsigned long localMemorySize;
	bool deviceAvailable;
	bool compilerAvailable;
	string deviceName;
	string deviceVendor;
	string driverVersion;
	string deviceProfile;
	string deviceVersion;
	string deviceOpenCLVersion;
	string deviceExtensions;

public:
	OpenCLDeviceInfo();
	~OpenCLDeviceInfo();
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICEINFO_H_ */
