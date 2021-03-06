/*
 * OCLHelper.cpp
 *
 *  Created on: Apr 27, 2015
 *      Author: Mikael
 */

#include "OCLHelper.h"
#include "OCLUtility.h"
#include <stdexcept>

namespace Matuna
{
namespace Helper
{

OCLPlatformInfo OCLHelper::GetPlatformInfo(
		cl_platform_id platformID)
{
	const size_t bufferSize = 10000;
	vector<char> charBuffer;
	charBuffer.resize(bufferSize);

	CheckOCLError(
			clGetPlatformInfo(platformID, CL_PLATFORM_NAME, bufferSize,
					charBuffer.data(), NULL),
			"We culd not retreive the platform name");
	string platformName(charBuffer.data());
	CheckOCLError(
			clGetPlatformInfo(platformID, CL_PLATFORM_PROFILE, bufferSize,
					charBuffer.data(), NULL),
			"Could not get the platform profile");
	string platformProfile(charBuffer.data());
	CheckOCLError(
			clGetPlatformInfo(platformID, CL_PLATFORM_VERSION, bufferSize,
					charBuffer.data(), NULL),
			"Could not get the platform version");
	string platformVersion(charBuffer.data());
	CheckOCLError(
			clGetPlatformInfo(platformID, CL_PLATFORM_VENDOR, bufferSize,
					charBuffer.data(), NULL),
			"Could not get the platform vendor");
	string platformVendor(charBuffer.data());
	CheckOCLError(
			clGetPlatformInfo(platformID, CL_PLATFORM_EXTENSIONS, bufferSize,
					charBuffer.data(), NULL),
			"Could not get the platform extensions");
	string platformExtensions(charBuffer.data());

	return OCLPlatformInfo(platformID, platformName, platformProfile,
			platformVersion, platformVendor, platformExtensions);
}

OCLDeviceInfo OCLHelper::GetDeviceInfo(
		const OCLPlatformInfo& platformInfo, cl_device_id deviceID)
{
	cl_device_type type;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_TYPE, sizeof(cl_device_type),
					&type, NULL), "Could not fetch the device type");
	cl_uint vendorID;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_VENDOR_ID, sizeof(cl_uint),
					&vendorID, NULL), "Could not fetch the vendor ID");
	cl_uint maxComputeUnits;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS,
					sizeof(cl_uint), &maxComputeUnits, NULL),
			"Could not fetch the max compute units");
	cl_uint maxWorkItemDimensions;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
					sizeof(cl_uint), &maxWorkItemDimensions, NULL),
			"Could not fetch the max work item dimension");
	cl_uint maxClockFrequency;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY,
					sizeof(cl_uint), &maxClockFrequency, NULL),
			"Could not fetch the max clock frequency");
	cl_uint preferredCharVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
					sizeof(cl_uint), &preferredCharVectorWidth, NULL),
			"Could not fetch the preferred char vector width");
	cl_uint preferredShortVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
					sizeof(cl_uint), &preferredShortVectorWidth, NULL),
			"Could not fetch the preferred short vector width");
	cl_uint preferredIntVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
					sizeof(cl_uint), &preferredIntVectorWidth, NULL),
			"Could not fetch the preferred int vector width");
	cl_uint preferredLongVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
					sizeof(cl_uint), &preferredLongVectorWidth, NULL),
			"Could not fetch the preferred long vector width");
	cl_uint preferredFloatVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
					sizeof(cl_uint), &preferredFloatVectorWidth, NULL),
			"Could not fetch the preferred float vector width");
	cl_uint preferredDoubleVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
					sizeof(cl_uint), &preferredDoubleVectorWidth, NULL),
			"Could not fetch the preferred double vector width");
	cl_uint preferredHalfVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
					sizeof(cl_uint), &preferredHalfVectorWidth, NULL),
			"Could not fetch the preferred half vector width");
	cl_uint nativeCharVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
					sizeof(cl_uint), &nativeCharVectorWidth, NULL),
			"Could not fetch the native char vector width");
	cl_uint nativeShortVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
					sizeof(cl_uint), &nativeShortVectorWidth, NULL),
			"Could not fetch the native short vector width");
	cl_uint nativeIntVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
					sizeof(cl_uint), &nativeIntVectorWidth, NULL),
			"Could not fetch the native int vector width");
	cl_uint nativeLongVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
					sizeof(cl_uint), &nativeLongVectorWidth, NULL),
			"Could not fetch the native long vector width");
	cl_uint nativeFloatVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
					sizeof(cl_uint), &nativeFloatVectorWidth, NULL),
			"Could not fetch the native float vector width");
	cl_uint nativeDoubleVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
					sizeof(cl_uint), &nativeDoubleVectorWidth, NULL),
			"Could not fetch the native double vector width");
	cl_uint nativeHalfVectorWidth;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
					sizeof(cl_uint), &nativeHalfVectorWidth, NULL),
			"Could not fetch the native half vector width");

	vector<size_t> maxWorkItemSizes;
	maxWorkItemSizes.resize(maxWorkItemDimensions);
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES,
					sizeof(size_t) * maxWorkItemDimensions,
					maxWorkItemSizes.data(), NULL),
			"Could not fetch the work item sizes");

	size_t maxWorkGroupSize;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE,
					sizeof(size_t), &maxWorkGroupSize, NULL),
			"Could not fetch the max work group size");
	cl_ulong maxMemoryAllocationSize;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
					sizeof(cl_ulong), &maxMemoryAllocationSize, NULL),
			"Could not fetch the max work group size");
	cl_bool imageSupport;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool),
					&imageSupport, NULL), "Could not fetch the image support");
	size_t maxParametersSize;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_PARAMETER_SIZE,
					sizeof(size_t), &maxParametersSize, NULL),
			"Could not fetch the max parameter size");
	cl_device_mem_cache_type globalMemoryCacheType;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
					sizeof(cl_device_mem_cache_type), &globalMemoryCacheType,
					NULL), "Could not fetch the global cache type");
	cl_ulong globalMemoryCacheSize;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
					sizeof(cl_ulong), &globalMemoryCacheSize, NULL),
			"Could not fetch the global memory cache size");
	cl_ulong globalMemorySize;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE,
					sizeof(cl_ulong), &globalMemorySize, NULL),
			"Could not fetch the global memory size");
	cl_ulong maxConstantBufferSize;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
					sizeof(cl_ulong), &maxConstantBufferSize, NULL),
			"Could not fetch the max constant buffer size");
	cl_uint maxConstantArguments;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_MAX_CONSTANT_ARGS,
					sizeof(cl_uint), &maxConstantArguments, NULL),
			"Could not fetch the max constant arguments count");
	cl_device_local_mem_type localMemoryType;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_TYPE,
					sizeof(cl_device_local_mem_type), &localMemoryType, NULL),
			"Could not fetch the local memory type");
	cl_ulong localMemorySize;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE,
					sizeof(cl_ulong), &localMemorySize, NULL),
			"Could not fetch the local memory size");
	cl_bool deviceAvailable;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_AVAILABLE, sizeof(cl_bool),
					&deviceAvailable, NULL),
			"Could not determine if the device is available");
	cl_bool compilerAvailable;
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_COMPILER_AVAILABLE,
					sizeof(cl_bool), &compilerAvailable, NULL),
			"Could not determine if a compiler is available");

	const size_t bufferSize = 10000;
	vector<char> charBuffer;
	charBuffer.resize(bufferSize);

	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_NAME, bufferSize,
					charBuffer.data(), NULL),
			"Could not fetch the device name");
	string deviceName(charBuffer.data());
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_VENDOR, bufferSize,
					charBuffer.data(), NULL),
			"Could not fetch the device vendor");
	string deviceVendor(charBuffer.data());
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DRIVER_VERSION, bufferSize,
					charBuffer.data(), NULL),
			"Could not fetch the device version");
	string driverVersion(charBuffer.data());
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_PROFILE, bufferSize,
					charBuffer.data(), NULL),
			"Could not fetch the device profile");
	string deviceProfile(charBuffer.data());
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, bufferSize,
					charBuffer.data(), NULL),
			"Could not fetch the device version");
	string deviceVersion(charBuffer.data());
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_VERSION, bufferSize,
					charBuffer.data(), NULL),
			"Could not fetch the device OCL C version");
	string deviceOCLVersion(charBuffer.data());
	CheckOCLError(
			clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, bufferSize,
					charBuffer.data(), NULL),
			"Could not fetch the device extensions");
	string deviceExtensions(charBuffer.data());

	return OCLDeviceInfo(platformInfo, deviceID, type, globalMemoryCacheType,
			localMemoryType, vendorID, maxComputeUnits, maxWorkItemDimensions,
			maxClockFrequency, maxWorkItemSizes, maxWorkGroupSize,
			preferredCharVectorWidth, preferredShortVectorWidth,
			preferredIntVectorWidth, preferredLongVectorWidth,
			preferredFloatVectorWidth, preferredDoubleVectorWidth,
			preferredHalfVectorWidth, nativeCharVectorWidth,
			nativeShortVectorWidth, nativeIntVectorWidth, nativeLongVectorWidth,
			nativeFloatVectorWidth, nativeDoubleVectorWidth,
			nativeHalfVectorWidth, maxMemoryAllocationSize, imageSupport != 0,
			maxParametersSize, globalMemoryCacheSize, globalMemorySize,
			maxConstantBufferSize, maxConstantArguments, localMemorySize,
			deviceAvailable != 0, compilerAvailable != 0, deviceName,
			deviceVendor, driverVersion, deviceProfile, deviceVersion,
			deviceOCLVersion, deviceExtensions);
}

vector<OCLPlatformInfo> OCLHelper::GetPlatformInfos()
{

	vector<OCLPlatformInfo> result;

	cl_uint platformCount;
	CheckOCLError(clGetPlatformIDs(0, NULL, &platformCount),
			"Could not get the number of platforms");

	vector<cl_platform_id> platforms;
	platforms.resize(platformCount);
	CheckOCLError(clGetPlatformIDs(platformCount, platforms.data(), NULL),
			"Could not retrive the platforms");

	for (auto platform : platforms)
		result.push_back(GetPlatformInfo(platform));

	return result;
}

vector<OCLDeviceInfo> OCLHelper::GetDeviceInfos(
		const OCLPlatformInfo& platformInfo)
{
	vector<OCLDeviceInfo> result;

	cl_uint deviceCount;
	CheckOCLError(
			clGetDeviceIDs(platformInfo.PlatformID(), CL_DEVICE_TYPE_ALL, 0,
			NULL, &deviceCount), "Could not fetch the device count");

	vector<cl_device_id> devices;
	devices.resize(deviceCount);
	CheckOCLError(
			clGetDeviceIDs(platformInfo.PlatformID(), CL_DEVICE_TYPE_ALL,
					deviceCount, devices.data(), NULL),
			"Could not fetch the devices");
	for (auto device : devices)
		result.push_back(GetDeviceInfo(platformInfo, device));

	return result;
}

unique_ptr<OCLContext> OCLHelper::GetContext(
		const OCLPlatformInfo& platformInfo, int queuesPerDevice,
		cl_command_queue_properties queueType)
{

	if (queuesPerDevice <= 0)
		throw invalid_argument(
				"We must have at least one command queue for a device");

	auto deviceInfos = GetDeviceInfos(platformInfo);
	vector<tuple<OCLDeviceConfig, OCLDeviceInfo>> deviceConfigs;

	for (auto& deviceInfo : deviceInfos)
	{
		OCLDeviceConfig deviceConfig;
		for (int i = 0; i < queuesPerDevice; i++)
			deviceConfig.AddCommandQueue(queueType);
		deviceConfigs.push_back(make_tuple(deviceConfig, deviceInfo));
	}

	return unique_ptr<OCLContext>(new OCLContext(platformInfo,deviceConfigs));
}

unique_ptr<OCLContext> OCLHelper::GetContext(const OCLPlatformInfo& platformInfo,
		vector<tuple<OCLDeviceConfig, OCLDeviceInfo>> deviceConfigs)
{
	for (auto& config : deviceConfigs)
		if (get<0>(config).CommandQueueCount() <= 0)
			throw invalid_argument(
					"We must have at least one command queue for a device");

	return unique_ptr<OCLContext>(new OCLContext(platformInfo, deviceConfigs));
}

} /* namespace Helper */
} /* namespace Matuna */
