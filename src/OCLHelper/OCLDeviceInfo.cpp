/*
 * OCLDeviceInfo.cpp
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#include "OCLDeviceInfo.h"
#include <sstream>
#include <stdexcept>

namespace Matuna {
	namespace Helper {

		OCLDeviceInfo::OCLDeviceInfo(
			OCLPlatformInfo platformInfo,
			cl_device_id deviceID,
			cl_device_type type,
			cl_device_mem_cache_type globalMemoryCacheType,
			cl_device_local_mem_type localMemoryType,
			unsigned int vendorID,
			unsigned int maxComputeUnits,
			unsigned int maxWorkItemDimensions,
			unsigned int maxClockFrequency,
			vector<size_t> maxWorkItemSizes,
			size_t maxWorkGroupSize,
			unsigned int preferredCharVectorWidth,
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
			unsigned long long maxMemoryAllocationSize, 
			bool imageSupport,
			size_t maxParametersSize, 
			unsigned long long globalMemoryCacheSize,
			unsigned long long globalMemorySize,
			unsigned long long maxConstantBufferSize,
			unsigned int maxConstantArguments,
			unsigned long long localMemorySize,
			bool deviceAvailable, 
			bool compilerAvailable,
			string deviceName,
			string deviceVendor, 
			string driverVersion, 
			string deviceProfile,
			string deviceVersion, 
			string deviceOCLVersion,
			string deviceExtensions) :
		platformInfo(platformInfo), deviceID(deviceID), globalMemoryCacheType(
			globalMemoryCacheType), localMemoryType(localMemoryType), type(
			type), vendorID(vendorID), maxComputeUnits(maxComputeUnits), maxWorkItemDimensions(
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
			maxParametersSize), globalMemoryCacheSize(
			globalMemoryCacheSize), globalMemorySize(globalMemorySize), maxConstantBufferSize(
			maxConstantBufferSize), maxConstantArguments(
			maxConstantArguments), localMemorySize(localMemorySize), deviceAvailable(
			deviceAvailable), compilerAvailable(compilerAvailable), deviceName(
			deviceName), deviceVendor(deviceVendor), driverVersion(
			driverVersion), deviceProfile(deviceProfile), deviceVersion(
			deviceVersion), deviceOCLVersion(deviceOCLVersion), deviceExtensions(
			deviceExtensions) {

		}

		OCLDeviceInfo::~OCLDeviceInfo() {

		}
		string OCLDeviceInfo::GetString() const {
			stringstream stringStream;
			stringStream << "PLATFORM INFORMATION: \n" << platformInfo.GetString()
				<< "\n";
			stringStream << "DEVICE INFORMATION: " << "\n";
			if (type == CL_DEVICE_TYPE_GPU)
				stringStream << "CL_DEVICE_TYPE: \t" << "GPU" << "\n";
			else if (type == CL_DEVICE_TYPE_CPU)
				stringStream << "CL_DEVICE_TYPE: \t" << "CPU" << "\n";
			else if (type == CL_DEVICE_TYPE_ACCELERATOR)
				stringStream << "CL_DEVICE_TYPE: \t" << "ACCELERATOR" << "\n";
			else if (type == CL_DEVICE_TYPE_DEFAULT)
				stringStream << "CL_DEVICE_TYPE: \t" << "DEFAULT" << "\n";
			else
				throw runtime_error("The device type is not exclusively specified");

			stringStream << "CL_DEVICE_VENDOR_ID: \t" << vendorID << "\n";
			stringStream << "CL_DEVICE_MAX_COMPUTE_UNITS: \t" << maxComputeUnits
				<< "\n";
			stringStream << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: \t"
				<< maxWorkItemDimensions << "\n";
			stringStream << "CL_DEVICE_MAX_CLOCK_FREQUENCY: \t" << maxClockFrequency
				<< "\n";
			stringStream << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: \t"
				<< preferredCharVectorWidth << "\n";
			stringStream << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: \t"
				<< preferredShortVectorWidth << "\n";
			stringStream << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: \t"
				<< preferredIntVectorWidth << "\n";
			stringStream << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: \t"
				<< preferredLongVectorWidth << "\n";
			stringStream << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: \t"
				<< preferredFloatVectorWidth << "\n";
			stringStream << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: \t"
				<< preferredDoubleVectorWidth << "\n";
			stringStream << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: \t"
				<< nativeHalfVectorWidth << "\n";
			stringStream << "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: \t"
				<< nativeCharVectorWidth << "\n";
			stringStream << "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: \t"
				<< nativeShortVectorWidth << "\n";
			stringStream << "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: \t"
				<< nativeIntVectorWidth << "\n";
			stringStream << "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: \t"
				<< nativeLongVectorWidth << "\n";
			stringStream << "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: \t"
				<< nativeFloatVectorWidth << "\n";
			stringStream << "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: \t"
				<< nativeDoubleVectorWidth << "\n";
			stringStream << "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF: \t"
				<< nativeHalfVectorWidth << "\n";

			if (maxWorkItemDimensions != maxWorkItemSizes.size())
				throw runtime_error(
				"The max work item sizes doesn't not have the same dimension as the maximum allowed dimension.");

			stringStream << "CL_DEVICE_MAX_WORK_ITEM_SIZES: \t";
			for (size_t i = 0; i < maxWorkItemDimensions; i++)
				stringStream << " (" << i << "): " << maxWorkItemSizes[i];
			stringStream << "\n";

			stringStream << "CL_DEVICE_MAX_WORK_GROUP_SIZE: \t" << maxWorkGroupSize
				<< "\n";
			stringStream << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: \t"
				<< maxMemoryAllocationSize << "\n";
			if (imageSupport == CL_TRUE)
				stringStream << "CL_DEVICE_IMAGE_SUPPORT: \t" << "TRUE" << "\n";
			else if (imageSupport == CL_FALSE)
				stringStream << "CL_DEVICE_IMAGE_SUPPORT: \t" << "FALSE" << "\n";
			else
				throw runtime_error(
				"It must be either true or false for image support");

			stringStream << "CL_DEVICE_MAX_PARAMETER_SIZE: \t" << maxParametersSize
				<< "\n";
			if (globalMemoryCacheType == CL_NONE)
				stringStream << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: \t" << "NONE" << "\n";
			else if (globalMemoryCacheType == CL_READ_ONLY_CACHE)
				stringStream << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: \t"
				<< "READ ONLY CACHE" << "\n";
			else if (globalMemoryCacheType == CL_READ_WRITE_CACHE)
				stringStream << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: \t"
				<< "READ WRITE CACHE" << "\n";

			stringStream << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: \t"
				<< globalMemoryCacheSize << "\n";
			stringStream << "CL_DEVICE_GLOBAL_MEM_SIZE: \t" << globalMemorySize << "\n";
			stringStream << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: \t"
				<< maxConstantBufferSize << "\n";
			stringStream << "CL_DEVICE_MAX_CONSTANT_ARGS: \t" << maxConstantArguments
				<< "\n";

			if (localMemoryType == CL_LOCAL)
				stringStream << "CL_DEVICE_LOCAL_MEM_TYPE: \t" << "LOCAL" << "\n";
			else if (localMemoryType == CL_GLOBAL)
				stringStream << "CL_DEVICE_LOCAL_MEM_TYPE: \t" << "GLOBAL" << "\n";
			else
				stringStream << "CL_DEVICE_LOCAL_MEM_TYPE: \t" << "NONE" << "\n";

			stringStream << "CL_DEVICE_LOCAL_MEM_SIZE: \t" << localMemorySize << "\n";

			if (deviceAvailable == CL_TRUE)
				stringStream << "CL_DEVICE_AVAILABLE: \t" << "TRUE" << "\n";
			else if (deviceAvailable == CL_FALSE)
				stringStream << "CL_DEVICE_AVAILABLE: \t" << "FALSE" << "\n";
			else
				throw runtime_error(
				"It must be either true or false for device availability");

			if (compilerAvailable == CL_TRUE)
				stringStream << "CL_DEVICE_COMPILER_AVAILABLE: \t" << "TRUE" << "\n";
			else if (compilerAvailable == CL_FALSE)
				stringStream << "CL_DEVICE_COMPILER_AVAILABLE: \t" << "FALSE" << "\n";
			else
				throw runtime_error(
				"It must be either true or false for compiler availability");

			stringStream << "CL_DEVICE_NAME: \t" << deviceName << "\n";
			stringStream << "CL_DEVICE_VENDOR: \t" << deviceVendor << "\n";
			stringStream << "CL_DRIVER_VERSION: \t" << driverVersion << "\n";
			stringStream << "CL_DEVICE_PROFILE: \t" << deviceProfile << "\n";
			stringStream << "CL_DEVICE_VERSION: \t" << deviceVersion << "\n";
			stringStream << "CL_DEVICE_OCL_C_VERSION: \t" << deviceOCLVersion
				<< "\n";
			stringStream << "CL_DEVICE_EXTENSIONS: \t" << deviceExtensions << "\n";

			return stringStream.str();
		}

	} /* namespace Helper */
} /* namespace Matuna */
