/*
 * OpenCLDeviceInfo.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 *
 * NOTE: Most of the comments here are directly taken from the OpenCL Specification.
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICEINFO_H_
#define ATML_OPENCLHELPER_OPENCLDEVICEINFO_H_

#include <vector>
#include <string>
#include <CL/cl.h>
#include "OpenCLPlatformInfo.h"

using namespace std;

namespace ATML {
	namespace Helper {

		/**
		*@brief The full description of an OpenCLDevice
		*
		*This class contains all information necessary in order to
		*tune and execute kernels on a particular OpenCLDevice.
		*
		*/
		class OpenCLDeviceInfo final{

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
			unsigned long long maxMemoryAllocationSize;
			bool imageSupport;
			size_t maxParametersSize;
			unsigned long long globalMemoryCacheSize;
			unsigned long long globalMemorySize;
			unsigned long long maxConstantBufferSize;
			unsigned int maxConstantArguments;
			unsigned long long localMemorySize;
			bool deviceAvailable;
			bool compilerAvailable;
			string deviceName;
			string deviceVendor;
			string driverVersion;
			string deviceProfile;
			string deviceVersion;
			string deviceOpenCLVersion;
			string deviceExtensions;

			/**
			*@brief Instantiate the OpenCLDeviceInfo.
			*@param platformInfo The PlatformInfo on which the device reside.
			*@param deviceID The native ID of the device.
			*@param type CL_DEVICE_TYPE
			*@param globalMemoryCacheType CL_DEVICE_GLOBAL_MEM_CACHE_TYPE
			*@param	localMemoryType CL_DEVICE_LOCAL_MEM_TYPE
			*@param vendorID CL_DEVICE_VENDOR_ID
			*@param maxComputeUnits CL_DEVICE_MAX_COMPUTE_UNITS
			*@param maxWorkItemDimensions CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
			*@param maxClockFrequency CL_DEVICE_MAX_CLOCK_FREQUENCY
			*@param maxWorkItemSizes CL_DEVICE_MAX_WORK_ITEM_SIZES
			*@param maxWorkGroupSize CL_DEVICE_MAX_WORK_GROUP_SIZE
			*@param preferredCharVectorWidth CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
			*@param preferredShortVectorWidth CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
			*@param preferredIntVectorWidth CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT
			*@param preferredLongVectorWidth CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG
			*@param preferredFloatVectorWidth CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
			*@param preferredDoubleVectorWidth CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
			*@param preferredHalfVectorWidth CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
			*@param nativeCharVectorWidth CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
			*@param nativeShortVectorWidth CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
			*@param nativeIntVectorWidth CL_DEVICE_NATIVE_VECTOR_WIDTH_INT
			*@param nativeLongVectorWidth CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
			*@param nativeFloatVectorWidth CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
			*@param nativeDoubleVectorWidth CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
			*@param nativeHalfVectorWidth CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
			*@param maxMemoryAllocationSize  CL_DEVICE_MAX_MEM_ALLOC_SIZE
			*@param imageSupport CL_DEVICE_IMAGE_SUPPORT
			*@param maxParametersSize CL_DEVICE_MAX_PARAMETER_SIZE
			*@param globalMemoryCacheSize CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
			*@param globalMemorySize CL_DEVICE_GLOBAL_MEM_SIZE
			*@param maxConstantBufferSize CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
			*@param maxConstantArguments CL_DEVICE_MAX_CONSTANT_ARGS
			*@param localMemorySize CL_DEVICE_LOCAL_MEM_SIZE
			*@param deviceAvailable CL_DEVICE_AVAILABLE
			*@param compilerAvailable CL_DEVICE_COMPILER_AVAILABLE
			*@param deviceName CL_DEVICE_NAME
			*@param deviceVendor CL_DEVICE_VENDOR
			*@param driverVersion CL_DRIVER_VERSION
			*@param deviceProfile CL_DEVICE_PROFILE
			*@param deviceVersion CL_DEVICE_VERSION
			*@param deviceOpenCLVersion CL_DEVICE_OPENCL_C_VERSION
			*@param deviceExtensions CL_DEVICE_EXTENSIONS
			*/
		public:
			OpenCLDeviceInfo(OpenCLPlatformInfo platformInfo,
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
				string deviceOpenCLVersion,
				string deviceExtensions);
			~OpenCLDeviceInfo();


			/**
			*@brief A string containing all the information about the device in a readable format.
			*
			*A string that summarized all of the information of this device in a human readable format.
			*
			*@return The string with the information.
			*/
			string GetString() const;

			/**
			*@brief platformInfo The PlatformInfo on which the device reside
			*
			*See OpenCLPlatformInfo
			*
			*@return the PlatformInfo of this device
			*/
			OpenCLPlatformInfo PlatformInfo() const {
				return platformInfo;
			}
			;

			/**
			*@brief 
			*
			*The native ID to the device. Used internally inside OpenCLDevice in order to execute OpenCLKernel.
			*
			*@return the native device ID.
			*/
			cl_device_id DeviceID() const {
				return deviceID;
			}
			;

			/**
			*@brief CL_DEVICE_TYPE
			*
			*CL_DEVICE_TYPE_CPU	An OpenCL device that is the host processor. The host processor runs the OpenCL implementations and is a single or multi-core CPU.
			*CL_DEVICE_TYPE_GPU	An OpenCL device that is a GPU. By this we mean that the device can also be used to accelerate a 3D API such as OpenGL or DirectX.
			*CL_DEVICE_TYPE_ACCELERATOR	Dedicated OpenCL accelerators (for example the IBM CELL Blade). These devices communicate with the host processor using a peripheral interconnect such as PCIe.
			*CL_DEVICE_TYPE_DEFAULT	The default OpenCL device in the system.
			*CL_DEVICE_TYPE_ALL	All OpenCL devices available in the system.
			*
			*@return the device type
			*/
			cl_device_type Type() const {
				return type;
			}
			;

			/**
			*@brief  CL_DEVICE_GLOBAL_MEM_CACHE_TYPE
			*
			*Type of global memory cache supported. Valid values are:
			*CL_NONE,
			*CL_READ_ONLY_CACHE and CL_READ_WRITE_CACHE.
			*
			*@return the global memory cache type
			*/
			cl_device_mem_cache_type GlobalMemoryCacheType() const {
				return globalMemoryCacheType;
			}
			;

			/**
			*@brief CL_DEVICE_LOCAL_MEM_TYPE
			*
			*Type of local memory supported. This can be set to CL_LOCAL implying dedicated local memory storage such as SRAM, or CL_GLOBAL.
			*For custom devices, CL_NONE can also be returned indicating no local memory support.
			*
			*@return the local memory type
			*/
			cl_device_local_mem_type LocalMemoryType() const {
				return localMemoryType;
			}
			;

			/**
			*@brief CL_DEVICE_VENDOR_ID
			*
			*
			*
			*@return the vendor ID.
			*/
			unsigned int VendorID() const {
				return vendorID;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_COMPUTE_UNITS
			*
			*The number of parallel compute units on the OpenCL device
			*
			*@return the amount of compute units.
			*/
			unsigned int MaxComputeUnits() const {
				return maxComputeUnits;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
			*
			*Maximum dimensions of the global and local ids
			*
			*@return the amount of work item dimensions supported by the device.
			*/
			unsigned int MaxWorkItemDimensions() const {
				return maxWorkItemDimensions;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_CLOCK_FREQUENCY
			*
			*Maximum configured clock frequency of the device in MHz.
			*
			*@return the maximum clock frequency.
			*/
			unsigned int MaxClockFrequency() const {
				return maxClockFrequency;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_WORK_ITEM_SIZES
			*
			*Maximum number of work item that can be specified in each dimension
			*
			*@return the maximum size in each supported dimension.
			*/
			vector<size_t> MaxWorkItemSizes() const {
				return maxWorkItemSizes;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_WORK_GROUP_SIZE
			*
			*Maximum number of of work-items in a work-group that a device is capable of executing on a single compute unit
			*
			*@return the maximum amount of work units on a single compute unit.
			*/
			size_t MaxWorkGroupSize() const {
				return maxWorkGroupSize;
			}
			;

			/**
			*@brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
			*
			*Preferred native vector width size for built-in scalar types that can be put into vectors.
			*
			*@return The width of the preferred vector.
			*/
			unsigned int PreferredCharVectorWidth() const {
				return preferredCharVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
			*
			*Preferred native vector width size for built-in scalar types that can be put into vectors.
			*
			*@return The width of the preferred vector.
			*/
			unsigned int PreferredShortVectorWidth() const {
				return preferredShortVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT
			*
			*Preferred native vector width size for built-in scalar types that can be put into vectors.
			*
			*@return The width of the preferred vector.
			*/
			unsigned int PreferredIntVectorWidth() const {
				return preferredIntVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG
			*
			*Preferred native vector width size for built-in scalar types that can be put into vectors.
			*
			*@return The width of the preferred vector.
			*/
			unsigned int PreferredLongVectorWidth() const {
				return preferredLongVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
			*
			*Preferred native vector width size for built-in scalar types that can be put into vectors.
			*
			*@return The width of the preferred vector.
			*/
			unsigned int PreferredFloatVectorWidth() const {
				return preferredFloatVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
			*
			*If double precision is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE must return 0.
			*
			*@return The width of the preferred vector.
			*/
			unsigned int PreferredDoubleVectorWidth() const {
				return preferredDoubleVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
			*
			*If the cl_khr_fp16 extension is not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF must return 0.
			*
			*@return The width of the preferred vector.
			*/
			unsigned int PreferredHalfVectorWidth() const {
				return preferredHalfVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
			*
			*Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
			*
			*@return The width of the native vector.
			*/
			unsigned int NativeCharVectorWidth() const {
				return nativeCharVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
			*
			*Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
			*
			*@return The width of the native vector.
			*/
			unsigned int NativeShortVectorWidth() const {
				return nativeShortVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_NATIVE_VECTOR_WIDTH_INT
			*
			*Returns the native ISA vector width.The vector width is defined as the number of scalar elements that can be stored in the vector.
			*
			*@return The width of the native vector.
			*/
			unsigned int NativeIntVectorWidth() const {
				return nativeIntVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
			*
			*Returns the native ISA vector width.The vector width is defined as the number of scalar elements that can be stored in the vector.
			*
			*@return The width of the native vector.
			*/
			unsigned int NativeLongVectorWidth() const {
				return nativeLongVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
			*
			*Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
			*
			*@return The width of the native vector.
			*/
			unsigned int NativeFloatVectorWidth() const {
				return nativeFloatVectorWidth;
			}
			;

			/**
			*@brief  CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
			*
			*If double precision is not supported, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE must return 0.
			*
			*@return The width of the native vector.
			*/
			unsigned int NativeDoubleVectorWidth() const {
				return nativeDoubleVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
			*
			*If the cl_khr_fp16 extension is not supported, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF must return 0.
			*
			*@return The width of the native vector.
			*/
			unsigned int NativeHalfVectorWidth() const {
				return nativeHalfVectorWidth;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_MEM_ALLOC_SIZE
			*
			*Max size of memory object allocation in bytes.
			*The minimum value is max (min(1024*1024*1024, 1/4th of CL_DEVICE_GLOBAL_MEM_SIZE), 32*1024*1024)
			*for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
			*
			*@return The maximum allocation size.
			*/
			unsigned long long MaxMemoryAllocationSize() const {
				return maxMemoryAllocationSize;
			}
			;

			/**
			*@brief CL_DEVICE_IMAGE_SUPPORT
			*
			*
			*
			*@return true for if the device supports images.
			*/
			bool ImageSupport() const {
				return imageSupport;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_PARAMETER_SIZE
			*
			*Max size in bytes of all arguments that can be passed to a kernel.
			*The minimum value is 1024 for devices that are not of type CL_DEVICE_TYPE_CUSTOM. For this minimum value, only a maximum of 128 arguments can be passed to a kernel.
			*
			*@return Max size in bytes of all arguments that can be passed to a kernel
			*/
			size_t MaxParametersSize() const {
				return maxParametersSize;
			}
			;

			/**
			*@brief CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
			*
			*Size of global memory cache in bytes.
			*
			*@return Size of global memory cache in bytes
			*/
			unsigned long long GlobalMemoryCacheSize() const {
				return globalMemoryCacheSize;
			}
			;

			/**
			*@brief CL_DEVICE_GLOBAL_MEM_SIZE
			*
			*Size of global device memory in bytes.
			*
			*@return Size of global device memory in bytes
			*/
			unsigned long long GlobalMemorySize() const {
				return globalMemorySize;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
			*
			*Max size in bytes of a constant buffer allocation. The minimum value is 64 KB for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
			*
			*@return Max size in bytes of a constant buffer allocation
			*/
			unsigned long long MaxConstantBufferSize() const {
				return maxConstantBufferSize;
			}
			;

			/**
			*@brief CL_DEVICE_MAX_CONSTANT_ARGS
			*
			*Max number of arguments declared with the __constant qualifier in a kernel. The minimum value is 8 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
			*
			*@return Max number of arguments declared with the __constant qualifier
			*/
			unsigned int MaxConstantArguments() const {
				return maxConstantArguments;
			}
			;

			/**
			*@brief CL_DEVICE_LOCAL_MEM_SIZE
			*
			*Size of local memory region in bytes. The minimum value is 32 KB for devices that are not of type CL_DEVICE_TYPE_CUSTOM.
			*
			*@return Size of local memory region in bytes
			*/
			unsigned long long LocalMemorySize() const {
				return localMemorySize;
			}
			;

			/**
			*@brief CL_DEVICE_AVAILABLE
			*
			*Is CL_TRUE if the device is available and CL_FALSE otherwise. A device is considered to be available if the device can be expected to successfully execute commands enqueued to the device.
			*
			*@return True if the device is available.
			*/
			bool DeviceAvailable() const {
				return deviceAvailable;
			}
			;

			/**
			*@brief CL_DEVICE_COMPILER_AVAILABLE
			*
			*Is CL_FALSE if the implementation does not have a compiler available to compile the program source.
			*Is CL_TRUE if the compiler is available.
			*This can be CL_FALSE for the embedded platform profile only.
			*
			*@return True if the compiler is available.
			*/
			bool CompilerAvailable() const {
				return compilerAvailable;
			}
			;

			/**
			*@brief  CL_DEVICE_NAME
			*
			*
			*
			*@return The device name as a human readable string.
			*/
			string DeviceName() const {
				return deviceName;
			}
			;

			/**
			*@brief CL_DEVICE_VENDOR
			*
			*
			*
			*@return The device vendor as a human readable string.
			*/
			string DeviceVendor() const {
				return deviceVendor;
			}
			;

			/**
			*@brief CL_DRIVER_VERSION
			*
			*OpenCL software driver version string in the form major_number.minor_number
			*
			*@return OpenCL software driver version string in the form major_number.minor_number
			*/
			string DriverVersion() const {
				return driverVersion;
			}
			;

			/**
			*@brief CL_DEVICE_PROFILE
			*
			*OpenCL profile string. Returns the profile name supported by the device. The profile name returned can be one of the following strings:
			*FULL_PROFILE – if the device supports the OpenCL specification (functionality defined as part of the core specification and
			*does not require any extensions to be supported).
			*EMBEDDED_PROFILE - if the device supports the OpenCL embedded profile.
			*
			*@return FULL_PROFILE or EMBEDDED_PROFILE
			*/
			string DeviceProfile() const {
				return deviceProfile;
			}
			;

			/**
			*@brief CL_DEVICE_VERSION
			*
			*OpenCL version string. Returns the OpenCL version supported by the device.
			*
			*@return OpenCL<space><major_version.minor_version><space><vendor-specific information>
			*/
			string DeviceVersion() const {
				return deviceVersion;
			}
			;

			/**
			*@brief CL_DEVICE_OPENCL_C_VERSION
			*
			*OpenCL C version string. Returns the highest OpenCL C version supported by the compiler for this device that is not of type CL_DEVICE_TYPE_CUSTOM.
			*
			*@return OpenCL<space>C<space><major_version.minor_version><space><vendor-specific information>
			*/
			string DeviceOpenCLVersion() const {
				return deviceOpenCLVersion;
			}
			;

			/**
			*@brief CL_DEVICE_EXTENSIONS
			*
			*Returns a space separated list of extension names (the extension names themselves do not contain any spaces) supported by the device.
			*The list of extension names returned can be vendor supported extension.
			*
			*@return Space separated list of extensions.
			*/
			string DeviceExtensions() const {
				return deviceExtensions;
			}
			;
		};

	} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICEINFO_H_ */
