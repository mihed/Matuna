/*
 * BackConvolutionKernelTest.cpp
 *
 *  Created on: May 24, 2015
 *      Author: Mikael
 */


#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OCLHelper/OCLHelper.h"
#include "ConvNetOCL/BackConvolutionKernel.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace Matuna::Helper;
using namespace Matuna::Math;
using namespace Matuna::MachineLearning;


SCENARIO("Back propagating a fully connected layer with the convolution kernel")
{

	WHEN("Convolving with local memory and without padding, offset")
	{
		random_device tempDevice;
		mt19937 mt(tempDevice());
		uniform_int_distribution<int> localDimensionGenerator(1, 16); //To this, we need to add the filter width / height
		uniform_int_distribution<int> imageGenerator(1, 30);
		uniform_int_distribution<int> filterDimensionGenerator(1, 10);

		for (int dummy = 0; dummy < 5; dummy++)
		{
			auto platformInfos = OCLHelper::GetPlatformInfos();
			for (auto& platformInfo : platformInfos)
			{
				auto context = OCLHelper::GetContext(platformInfo);
				auto devices = context->GetDevices();
				for (auto device : devices)
				{
					vector<OCLDevice*> oneDeviceVector;
					oneDeviceVector.push_back(device);

					int filterWidth = filterDimensionGenerator(mt);
					int filterHeight = filterDimensionGenerator(mt);
					int filterUnits = filterDimensionGenerator(mt);

					int localWidth = localDimensionGenerator(mt);
					int localHeight = localDimensionGenerator(mt);

					int outputWidth = localWidth * imageGenerator(mt);
					int outputHeight = localHeight * imageGenerator(mt);

					int imageWidth = outputWidth + filterWidth - 1;
					int imageHeight = outputHeight + filterHeight - 1;

					vector<Matrixf> filters;
					vector<Matrixf> inputs;

					INFO("Creating random inputs and filters");
					for (int i = 0; i < filterUnits; i++)
					{
						filters.push_back(Matrixf::RandomNormal(filterHeight, filterWidth));
						inputs.push_back(Matrixf::RandomNormal(imageHeight, imageWidth));
					}

					INFO("Creating the back prop kernel");
					BackConvolutionKernel<float> kernel(outputWidth, outputHeight, filterUnits, filterWidth, filterHeight,
						0, 0, 0, 0, 0, imageWidth, outputWidth, imageHeight, true);

					kernel.InitializeCompilerOptions();

					INFO("Compiling and adding the kernel");
					context->AddProgramFromSource(&kernel, oneDeviceVector);
					context->AddKernel(&kernel);

					INFO("Creating the contigous memory for the OCL buffers");
					unique_ptr<float[]> filterBuffer(new float[filterUnits * filterWidth * filterHeight]);
					unique_ptr<float[]> inputBuffer(new float[filterUnits * imageWidth *  imageHeight]);
					float* rawFilterPointer = filterBuffer.get();
					float* rawInputPointer = inputBuffer.get();

					for (int i = 0; i < filterUnits; i++)
					{
						auto& filter = filters[i];
						auto& input = inputs[i];

						memcpy(rawFilterPointer, filter.Data, sizeof(float) * filter.ElementCount());
						memcpy(rawInputPointer, input.Data, sizeof(float) * input.ElementCount());

						rawFilterPointer += filter.ElementCount();
						rawInputPointer += input.ElementCount();
					}

					INFO("Creating the OCL buffers");
					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) *  imageWidth * imageHeight *  filterUnits);
					auto filterMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) *  filterWidth *  filterHeight *  filterUnits);
					auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(float) * outputWidth *  outputHeight);

					INFO("Writing the host buffers to the OCL memory");
					device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), inputBuffer.get());
					device->WriteMemory(filterMemory.get(), filterMemory->ByteSize(), filterBuffer.get());
					device->WaitForDeviceQueue(0);

					INFO("Setting the kernel arguments");
					kernel.SetDeltaInput(inputMemory.get());
					kernel.SetFilters(filterMemory.get());
					kernel.SetOutput(outputMemory.get());

					INFO("Setting the local memory size");
					auto maximumLocalMemory = device->DeviceInfo().LocalMemorySize();
					auto maximumLocalWorkGroupDimensions = device->DeviceInfo().MaxWorkItemSizes();

					//Ok, here we need to query the kernel info
					OCLKernelInfo kernelInfo = device->GetKernelInfo(&kernel);

					//Check that the local size is valid (I don't care about memory overflow here)
					auto maxWorkGroupSize = kernelInfo.KernelWorkGroupSize();
					while ((localWidth * localHeight) > maxWorkGroupSize || (outputWidth % localWidth != 0 || outputHeight % localHeight != 0))
					{
						localWidth--;
						localHeight--;

						if (localWidth <= 0)
							break;

						if (localHeight <= 0)
							break;

						if (outputWidth % localWidth != 0)
							continue;
						if (outputHeight % localHeight != 0)
							continue;
					}

					if (maximumLocalMemory < ((localWidth + filterWidth - 1)* (localHeight + filterHeight - 1) * filterUnits * sizeof(float)))
						WARN("We are using too much local memory. This will probably slow down the execution or fail.");

					if (localHeight < 1)
					{
						WARN("No fitting local height, skipping");
						continue;
					}

					if (localWidth < 1)
					{
						WARN("No fitting local width, skipping");
						continue;
					}

					auto temp = outputWidth % localWidth;
					CHECK(temp == 0);
					temp = outputHeight % localHeight;
					CHECK(temp == 0);

					kernel.SetLocalWorkGroup(localWidth, localHeight);

					INFO("Executing the kernel");
					device->ExecuteKernel(&kernel);
					device->WaitForDeviceQueue(0);

					INFO("Reading the OCL result");
					Matrixf kernelResult(outputHeight, outputWidth);
					device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), kernelResult.Data);
					device->WaitForDeviceQueue(0);

					//cout << "Kernel result: \n" << kernelResult.GetString() << endl;

					INFO("Calculating the manual result");
					Matrixf manualResult = Matrixf::Zeros(outputHeight, outputWidth);
					for (int i = 0; i < filterUnits; i++)
					{
						auto temp = inputs[i].Convolve(filters[i].Rotate180());
						CHECK(manualResult.RowCount() == temp.RowCount());
						CHECK(manualResult.ColumnCount() == temp.ColumnCount());
						manualResult += temp;
					}

					INFO("Comparing the kernel result to the manual result");
					auto difference = (kernelResult - manualResult).Norm2Square() / manualResult.ElementCount();
					cout << "Difference: " << difference << endl;
					CHECK(difference < 1E-5);
				}
			}
		}
	}

	WHEN("Convolving without padding, offset and local memory")
	{
		random_device tempDevice;
		mt19937 mt(tempDevice());
		uniform_int_distribution<int> dimensionGenerator(30, 1000);
		uniform_int_distribution<int> filterDimensionGenerator(1, 30);
		normal_distribution<float> distribution;

		for (int dummy = 0; dummy < 5; dummy++)
		{
			auto platformInfos = OCLHelper::GetPlatformInfos();
			for (auto& platformInfo : platformInfos)
			{
				auto context = OCLHelper::GetContext(platformInfo);
				auto devices = context->GetDevices();
				for (auto device : devices)
				{
					vector<OCLDevice*> oneDeviceVector;
					oneDeviceVector.push_back(device);

					int filterWidth = filterDimensionGenerator(mt);
					int filterHeight = filterDimensionGenerator(mt);
					int filterUnits = filterDimensionGenerator(mt);

					int imageWidth = dimensionGenerator(mt);
					int imageHeight = dimensionGenerator(mt);

					int outputWidth = imageWidth - filterWidth + 1;
					int outputHeight = imageHeight - filterHeight + 1;

					vector<Matrixf> filters;
					vector<Matrixf> inputs;

					INFO("Creating random inputs and filters");
					for (int i = 0; i < filterUnits; i++)
					{
						filters.push_back(Matrixf::RandomNormal(filterHeight, filterWidth));
						inputs.push_back(Matrixf::RandomNormal(imageHeight, imageWidth));
					}

					INFO("Creating the back prop kernel");
					BackConvolutionKernel<float> kernel(outputWidth, outputHeight, filterUnits, filterWidth, filterHeight,
						0, 0, 0, 0, 0, imageWidth, outputWidth, imageHeight);

					kernel.InitializeCompilerOptions();

					INFO("Compiling and adding the kernel");
					context->AddProgramFromSource(&kernel, oneDeviceVector);
					context->AddKernel(&kernel);

					INFO("Creating the contigous memory for the OCL buffers");
					unique_ptr<float[]> filterBuffer(new float[filterUnits * filterWidth * filterHeight]);
					unique_ptr<float[]> inputBuffer(new float[filterUnits * imageWidth *  imageHeight]);
					float* rawFilterPointer = filterBuffer.get();
					float* rawInputPointer = inputBuffer.get();

					for (int i = 0; i < filterUnits; i++)
					{
						auto& filter = filters[i];
						auto& input = inputs[i];

						memcpy(rawFilterPointer, filter.Data, sizeof(float) * filter.ElementCount());
						memcpy(rawInputPointer, input.Data, sizeof(float) * input.ElementCount());

						rawFilterPointer += filter.ElementCount();
						rawInputPointer += input.ElementCount();
					}

					INFO("Creating the OCL buffers");
					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(float) *  imageWidth * imageHeight *  filterUnits);
					auto filterMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof (float) *  filterWidth *  filterHeight *  filterUnits);
					auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(float) * outputWidth *  outputHeight);

					INFO("Writing the host buffers to the OCL memory");
					device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), inputBuffer.get());
					device->WriteMemory(filterMemory.get(), filterMemory->ByteSize(), filterBuffer.get());
					device->WaitForDeviceQueue(0);

					INFO("Setting the kernel arguments");
					kernel.SetDeltaInput(inputMemory.get());
					kernel.SetFilters(filterMemory.get());
					kernel.SetOutput(outputMemory.get());

					INFO("Executing the kernel");
					device->ExecuteKernel(&kernel);
					device->WaitForDeviceQueue(0);

					INFO("Reading the OCL result");
					Matrixf kernelResult(outputHeight, outputWidth);
					device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), kernelResult.Data);
					device->WaitForDeviceQueue(0);

					INFO("Calculating the manual result");
					Matrixf manualResult = Matrixf::Zeros(outputHeight, outputWidth);
					for (int i = 0; i < filterUnits; i++)
					{
						auto temp = inputs[i].Convolve(filters[i].Rotate180());
						CHECK(manualResult.RowCount() == temp.RowCount());
						CHECK(manualResult.ColumnCount() == temp.ColumnCount());
						manualResult += temp;
					}

					INFO("Comparing the kernel result to the manual result");
					auto difference = (kernelResult - manualResult).Norm2Square() / manualResult.ElementCount();
					cout << "Difference: " << difference << endl;
					CHECK(difference < 1E-5);
				}
			}
		}
	}
}