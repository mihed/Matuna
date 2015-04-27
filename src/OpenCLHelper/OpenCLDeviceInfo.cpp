/*
 * OpenCLDeviceInfo.cpp
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#include "OpenCLDeviceInfo.h"

namespace ATML {
namespace Helper {

OpenCLDeviceInfo::OpenCLDeviceInfo(OpenCLPlatformInfo platformInfo,
		cl_device_id deviceID, cl_device_type type,
		cl_device_mem_cache_type globalMemoryCacheType,
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
		unsigned int nativeCharVectorWidth, unsigned int nativeShortVectorWidth,
		unsigned int nativeIntVectorWidth,
		unsigned int nativeLongVectorWidth, unsigned int nativeFloatVectorWidth,
		unsigned int nativeDoubleVectorWidth,
		unsigned int nativeHalfVectorWidth,
		unsigned long maxMemoryAllocationSize, bool imageSupport,
		size_t maxParametersSize, unsigned long globalMemoryCacheSize,
		unsigned long globalMemorySize, unsigned long maxConstantBufferSize,
		unsigned int maxConstantArguments, unsigned long localMemorySize,
		bool deviceAvailable, bool compilerAvailable, string deviceName,
		string deviceVendor, string driverVersion, string deviceProfile,
		string deviceVersion, string deviceOpenCLVersion,
		string deviceExtensions) :
		platformInfo(platformInfo), deviceID(deviceID), type(type), vendorID(
				vendorID), maxComputeUnits(maxComputeUnits), maxWorkItemDimensions(
				maxWorkItemDimensions), maxWorkItemSizes(maxWorkItemSizes), maxWorkGroupSize(
				maxWorkGroupSize), maxClockFrequency(maxClockFrequency), preferredCharVectorWidth(
				preferredCharVectorWidth), preferredShortVectorWidth(
				preferredShortVectorWidth), preferredIntVectorWidth(
				preferredIntVectorWidth), preferredLongVectorWidth(
				preferredLongVectorWidth), preferredFloatVectorWidth(
				preferredFloatVectorWidth), preferredDoubleVectorWidth(
				preferredDoubleVectorWidth), preferredHalfVectorWidth(
				preferredHalfVectorWidth), nativeCharVectorWidth(
				nativeCharVectorWidth), nativeShortVectorWidth(
				nativeShortVectorWidth), nativeIntVectorWidth(
				nativeIntVectorWidth), nativeLongVectorWidth(
				nativeLongVectorWidth), nativeFloatVectorWidth(
				nativeFloatVectorWidth), nativeDoubleVectorWidth(
				nativeDoubleVectorWidth), nativeHalfVectorWidth(
				nativeHalfVectorWidth), maxMemoryAllocationSize(
				maxMemoryAllocationSize), imageSupport(imageSupport), maxParametersSize(
				maxParametersSize), globalMemoryCacheType(
				globalMemoryCacheType), globalMemoryCacheSize(
				globalMemoryCacheSize), globalMemorySize(globalMemorySize), maxConstantBufferSize(
				maxConstantBufferSize), maxConstantArguments(
				maxConstantArguments), localMemoryType(localMemoryType), localMemorySize(
				localMemorySize), deviceAvailable(deviceAvailable), compilerAvailable(
				compilerAvailable), deviceName(deviceName), deviceVendor(
				deviceVendor), driverVersion(driverVersion), deviceProfile(
				deviceProfile), deviceVersion(deviceVersion), deviceOpenCLVersion(
				deviceOpenCLVersion) {


}

OpenCLDeviceInfo::~OpenCLDeviceInfo() {

}

} /* namespace Helper */
} /* namespace ATML */
