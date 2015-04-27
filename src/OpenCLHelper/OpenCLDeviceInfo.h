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
	unsigned int nativeShortVectorWidth;
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
	OpenCLDeviceInfo(OpenCLPlatformInfo platformInfo, cl_device_id deviceID,
			cl_device_type type, cl_device_mem_cache_type globalMemoryCacheType,
			cl_device_local_mem_type localMemoryType, unsigned int vendorID,
			unsigned int maxComputeUnits, unsigned int maxWorkItemDimensions,
			unsigned int maxClockFrequency, vector<size_t> maxWorkItemSizes,
			size_t maxWorkGroupSize, unsigned int preferredCharVectorWidth,
			unsigned int preferredShortVectorWidth,
			unsigned int preferredIntVectorWidth,
			unsigned int preferredLongVectorWidth,
			unsigned int preferredFloatVectorWidth,
			unsigned int preferredDoubleVectorWidth,
			unsigned int preferredHalfVectorWidth,
			unsigned int nativeCharVectorWidth,
			unsigned int nativeShortVectorWidth,
			unsigned int nativeIntVectorWidth,
			unsigned int nativeLongVectorWidth,
			unsigned int nativeFloatVectorWidth,
			unsigned int nativeDoubleVectorWidth,
			unsigned int nativeHalfVectorWidth,
			unsigned long maxMemoryAllocationSize, bool imageSupport,
			size_t maxParametersSize, unsigned long globalMemoryCacheSize,
			unsigned long globalMemorySize, unsigned long maxConstantBufferSize,
			unsigned int maxConstantArguments, unsigned long localMemorySize,
			bool deviceAvailable, bool compilerAvailable, string deviceName,
			string deviceVendor, string driverVersion, string deviceProfile,
			string deviceVersion, string deviceOpenCLVersion,
			string deviceExtensions);
	~OpenCLDeviceInfo();

	OpenCLPlatformInfo PlatformInfo() const {
		return platformInfo;
	}
	;
	cl_device_id DeviceID() const {
		return deviceID;
	}
	;
	cl_device_type Type() const {
		return type;
	}
	;
	cl_device_mem_cache_type GlobalMemoryCacheType() const {
		return globalMemoryCacheType;
	}
	;
	cl_device_local_mem_type LocalMemoryType() const {
		return localMemoryType;
	}
	;

	unsigned int VendorID() const {
		return vendorID;
	}
	;
	unsigned int MaxComputeUnits() const {
		return maxComputeUnits;
	}
	;
	unsigned int MaxWorkItemDimensions() const {
		return maxWorkItemDimensions;
	}
	;
	unsigned int MaxClockFrequency() const {
		return maxClockFrequency;
	}
	;
	vector<size_t> MaxWorkItemSizes() const {
		return maxWorkItemSizes;
	}
	;
	size_t MaxWorkGroupSize() const {
		return maxWorkGroupSize;
	}
	;
	unsigned int PreferredCharVectorWidth() const {
		return preferredCharVectorWidth;
	}
	;
	unsigned int PreferredShortVectorWidth() const {
		return preferredShortVectorWidth;
	}
	;
	unsigned int PreferredIntVectorWidth() const {
		return preferredIntVectorWidth;
	}
	;
	unsigned int PreferredLongVectorWidth() const {
		return preferredLongVectorWidth;
	}
	;
	unsigned int PreferredFloatVectorWidth() const {
		return preferredFloatVectorWidth;
	}
	;
	unsigned int PreferredDoubleVectorWidth() const {
		return preferredDoubleVectorWidth;
	}
	;
	unsigned int PreferredHalfVectorWidth() const {
		return preferredHalfVectorWidth;
	}
	;
	unsigned int NativeCharVectorWidth() const {
		return nativeCharVectorWidth;
	}
	;
	unsigned int NativeShortVectorWidth() const {
		return nativeShortVectorWidth;
	}
	;
	unsigned int NativeIntVectorWidth() const {
		return nativeIntVectorWidth;
	}
	;
	unsigned int NativeLongVectorWidth() const {
		return nativeLongVectorWidth;
	}
	;
	unsigned int NativeFloatVectorWidth() const {
		return nativeFloatVectorWidth;
	}
	;
	unsigned int NativeDoubleVectorWidth() const {
		return nativeDoubleVectorWidth;
	}
	;
	unsigned int NativeHalfVectorWidth() const {
		return nativeHalfVectorWidth;
	}
	;
	unsigned long MaxMemoryAllocationSize() const {
		return maxMemoryAllocationSize;
	}
	;
	bool ImageSupport() const {
		return imageSupport;
	}
	;
	size_t MaxParametersSize() const {
		return maxParametersSize;
	}
	;
	unsigned long GlobalMemoryCacheSize() const {
		return globalMemoryCacheSize;
	}
	;
	unsigned long GlobalMemorySize() const {
		return globalMemorySize;
	}
	;
	unsigned long MaxConstantBufferSize() const {
		return maxConstantBufferSize;
	}
	;
	unsigned int MaxConstantArguments() const {
		return maxConstantArguments;
	}
	;
	unsigned long LocalMemorySize() const {
		return localMemorySize;
	}
	;
	bool DeviceAvailable() const {
		return deviceAvailable;
	}
	;
	bool CompilerAvailable() const {
		return compilerAvailable;
	}
	;
	string DeviceName() const {
		return deviceName;
	}
	;
	string DeviceVendor() const {
		return deviceVendor;
	}
	;
	string DriverVersion() const {
		return driverVersion;
	}
	;
	string DeviceProfile() const {
		return deviceProfile;
	}
	;
	string DeviceVersion() const {
		return deviceVersion;
	}
	;
	string DeviceOpenCLVersion() const {
		return deviceOpenCLVersion;
	}
	;
	string DeviceExtensions() const {
		return deviceExtensions;
	}
	;
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICEINFO_H_ */
